"""Contact-aided invariant EKF (Hartley, Ghaffari, Eustice, Grizzle, IJRR 2020; the Olaf paper's state estimator [42]).

Batched (one filter per env), right-invariant ("world-centric") form of the open-source invariant-ekf library:

  state X in SE_4(3): R (body -> world), v, p (body origin = IMU), d_L, d_R (sole contact points, world)
  parameters theta: gyro bias b_g, accelerometer bias b_a
  error order (21): [phi_R, v, p, d_L, d_R, b_g, b_a]

  propagate(gyro, accel, dt): strapdown IMU integration, covariance with the log-linear error dynamics
  correct(s, contact): forward-kinematics measurement s_i = R^T (d_i - p) of every sole in contact
  a sole that touches down is (re)initialised at p + R s_i with the covariance of p (library augmentation); a sole in
  the air stays in the state unused (the library removes it; a fixed size keeps it batched)

humanoid_controller (branch arti-h1-olaf) robot::OlafInEKF is the C++ twin of this class: same equations, same order.
float64 throughout.
"""

from __future__ import annotations

import torch

GRAVITY = (0.0, 0.0, -9.81)
DIM = 21
IR, IV, IP, ID, IBG, IBA = 0, 3, 6, 9, 15, 18


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack((torch.stack((z, -w, y), -1), torch.stack((w, z, -x), -1), torch.stack((-y, x, z), -1)), -2)


def so3_exp_jl(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exp(phi) and the left Jacobian J_l(phi) (Rodrigues; small-angle series below 1e-6 rad)."""
    th = phi.norm(dim=-1)[..., None, None]
    K = skew(phi)
    K2 = K @ K
    small = th < 1e-6
    ths = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, 1.0 - th * th / 6.0, torch.sin(ths) / ths)
    b = torch.where(small, 0.5 - th * th / 24.0, (1.0 - torch.cos(ths)) / (ths * ths))
    c = torch.where(small, 1.0 / 6.0 - th * th / 120.0, (ths - torch.sin(ths)) / (ths * ths * ths))
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(K)
    return eye + a * K + b * K2, eye + b * K + c * K2


class InEKF:
    def __init__(self, n: int, device, *, gyro_noise: float, accel_noise: float, gyro_bias_noise: float,
                 accel_bias_noise: float, contact_noise: float, fk_noise: float, init_std: dict[str, float]):
        f64 = torch.float64
        self.n, self.dev = n, device
        self.R = torch.eye(3, dtype=f64, device=device).repeat(n, 1, 1)
        self.v = torch.zeros(n, 3, dtype=f64, device=device)
        self.p = torch.zeros(n, 3, dtype=f64, device=device)
        self.d = torch.zeros(n, 2, 3, dtype=f64, device=device)
        self.bg = torch.zeros(n, 3, dtype=f64, device=device)
        self.ba = torch.zeros(n, 3, dtype=f64, device=device)
        self.P = torch.eye(DIM, dtype=f64, device=device).repeat(n, 1, 1)
        self.contact = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.g = torch.tensor(GRAVITY, dtype=f64, device=device)
        # continuous-time noise densities (std per sqrt(Hz)), body frame
        q = torch.zeros(DIM, dtype=f64, device=device)
        q[IR:IR + 3] = gyro_noise ** 2
        q[IV:IV + 3] = accel_noise ** 2
        q[ID:ID + 6] = contact_noise ** 2
        q[IBG:IBG + 3] = gyro_bias_noise ** 2
        q[IBA:IBA + 3] = accel_bias_noise ** 2
        self.Qc = torch.diag(q)
        self.fk_var = fk_noise ** 2
        s = init_std
        p0 = torch.zeros(DIM, dtype=f64, device=device)
        p0[IR:IR + 3] = s["rot"] ** 2
        p0[IV:IV + 3] = s["vel"] ** 2
        p0[IP:IP + 3] = s["pos"] ** 2
        p0[IBG:IBG + 3] = s["gyro_bias"] ** 2
        p0[IBA:IBA + 3] = s["accel_bias"] ** 2
        self.P0 = torch.diag(p0)
        self.eye = torch.eye(DIM, dtype=f64, device=device)

    # ------------------------------------------------------------------ reset
    def reset(self, mask: torch.Tensor, R0: torch.Tensor, s: torch.Tensor, ground_z: float = 0.0):
        """Start the filter at attitude R0 (body -> world), zero velocity / biases, the body origin above the soles s
        (body frame, (k, 2, 3)) with the lowest sole at ground_z (the other one may be in the air: mid-stride starts),
        x / y = 0. Both soles are augmented; the next correct() measures only the ones in contact."""
        if not mask.any():
            return
        R0 = R0.to(torch.float64)
        s = s.to(torch.float64)
        rel = R0[:, None] @ s[..., None]
        p = torch.zeros(R0.shape[0], 3, dtype=torch.float64, device=self.dev)
        p[:, 2] = ground_z - rel[:, :, 2, 0].min(1).values
        self.R[mask], self.v[mask], self.p[mask] = R0, 0.0, p
        self.bg[mask], self.ba[mask] = 0.0, 0.0
        self.P[mask] = self.P0
        self.contact[mask] = False
        both = torch.ones(R0.shape[0], 2, dtype=torch.bool, device=self.dev)
        idx = torch.nonzero(mask).flatten()
        self._augment(idx, both, s)
        self.contact[mask] = True

    # ------------------------------------------------------------------ propagation
    def propagate(self, gyro: torch.Tensor, accel: torch.Tensor, dt: float):
        gyro, accel = gyro.to(torch.float64), accel.to(torch.float64)
        R, v, p, d = self.R, self.v, self.p, self.d
        w = gyro - self.bg
        a = accel - self.ba
        # linearised right-invariant error dynamics at the current estimate
        n = self.n
        A = torch.zeros(n, DIM, DIM, dtype=torch.float64, device=self.dev)
        A[:, IV:IV + 3, IR:IR + 3] = skew(self.g).expand(n, 3, 3)
        A[:, IP:IP + 3, IV:IV + 3] = torch.eye(3, dtype=torch.float64, device=self.dev)
        A[:, IR:IR + 3, IBG:IBG + 3] = -R
        A[:, IV:IV + 3, IBG:IBG + 3] = -skew(v) @ R
        A[:, IP:IP + 3, IBG:IBG + 3] = -skew(p) @ R
        for i in range(2):
            A[:, ID + 3 * i:ID + 3 * i + 3, IBG:IBG + 3] = -skew(d[:, i]) @ R
        A[:, IV:IV + 3, IBA:IBA + 3] = -R
        Adt = A * dt
        Phi = self.eye + Adt + 0.5 * Adt @ Adt
        # noise mapped by the adjoint of X (biases: identity)
        Ad = torch.zeros(n, DIM, DIM, dtype=torch.float64, device=self.dev)
        Ad[:, IR:IR + 3, IR:IR + 3] = R
        Ad[:, IV:IV + 3, IR:IR + 3] = skew(v) @ R
        Ad[:, IV:IV + 3, IV:IV + 3] = R
        Ad[:, IP:IP + 3, IR:IR + 3] = skew(p) @ R
        Ad[:, IP:IP + 3, IP:IP + 3] = R
        for i in range(2):
            k = ID + 3 * i
            Ad[:, k:k + 3, IR:IR + 3] = skew(d[:, i]) @ R
            Ad[:, k:k + 3, k:k + 3] = R
        Ad[:, IBG:, IBG:] = torch.eye(6, dtype=torch.float64, device=self.dev)
        Q = Phi @ Ad @ self.Qc @ Ad.transpose(1, 2) @ Phi.transpose(1, 2) * dt
        P = Phi @ self.P @ Phi.transpose(1, 2) + Q
        self.P = 0.5 * (P + P.transpose(1, 2))
        # mean (strapdown, previous attitude for the acceleration)
        acc_w = (R @ a[..., None])[..., 0] + self.g
        self.p = p + v * dt + 0.5 * acc_w * dt * dt
        self.v = v + acc_w * dt
        self.R = R @ so3_exp_jl(w * dt)[0]

    # ------------------------------------------------------------------ correction
    def _augment(self, idx: torch.Tensor, new: torch.Tensor, s: torch.Tensor):
        """Touchdown of sole i (new[:, i]) in envs idx: d_i = p + R s_i, error of d_i = error of p (+ FK noise)."""
        for i in range(2):
            m = new[:, i]
            if not m.any():
                continue
            e = idx[m]
            R, p = self.R[e], self.p[e]
            self.d[e, i] = p + (R @ s[m, i][..., None])[..., 0]
            P = self.P[e]
            F = self.eye.expand(len(e), DIM, DIM).clone()
            k = ID + 3 * i
            F[:, k:k + 3, :] = 0.0
            F[:, k:k + 3, IP:IP + 3] = torch.eye(3, dtype=torch.float64, device=self.dev)
            P = F @ P @ F.transpose(1, 2)
            P[:, k:k + 3, k:k + 3] += self.fk_var * torch.eye(3, dtype=torch.float64, device=self.dev)
            self.P[e] = P

    def correct(self, s: torch.Tensor, contact: torch.Tensor):
        """Soles s (body frame, (n, 2, 3)) and their contact flags. New contacts are augmented, the others measured."""
        s = s.to(torch.float64)
        self.nis = torch.full((self.n, 2), float("nan"), dtype=torch.float64, device=self.dev)  # diagnostics
        new = contact & ~self.contact
        if new.any():
            idx = torch.arange(self.n, device=self.dev)
            rows = new.any(-1)
            self._augment(idx[rows], new[rows], s[rows])
        meas = contact & ~new
        for i in range(2):
            m = meas[:, i]
            if not m.any():
                continue
            k = ID + 3 * i
            H = torch.zeros(self.n, 3, DIM, dtype=torch.float64, device=self.dev)
            H[:, :, IP:IP + 3] = -torch.eye(3, dtype=torch.float64, device=self.dev)
            H[:, :, k:k + 3] = torch.eye(3, dtype=torch.float64, device=self.dev)
            Z = (self.R @ s[:, i][..., None])[..., 0] + self.p - self.d[:, i]
            PHt = self.P @ H.transpose(1, 2)
            S = H @ PHt + self.fk_var * torch.eye(3, dtype=torch.float64, device=self.dev)
            Sinv = torch.linalg.inv(S)
            self.nis[:, i] = torch.where(m, (Z[:, None] @ Sinv @ Z[..., None])[:, 0, 0], torch.full_like(Z[:, 0], float("nan")))
            K = PHt @ Sinv
            K = K * m[:, None, None]
            delta = (K @ Z[..., None])[..., 0]
            self._apply(delta)
            IKH = self.eye - K @ H
            P = IKH @ self.P @ IKH.transpose(1, 2) + self.fk_var * K @ K.transpose(1, 2)
            self.P = 0.5 * (P + P.transpose(1, 2))
        self.contact = contact.clone()

    def _apply(self, delta: torch.Tensor):
        """X <- Exp(delta_X) X (right-invariant update), theta <- theta + delta_theta."""
        E, J = so3_exp_jl(delta[:, IR:IR + 3])

        def upd(x, k):
            return (E @ x[..., None])[..., 0] + (J @ delta[:, k:k + 3][..., None])[..., 0]

        self.R = E @ self.R
        self.v = upd(self.v, IV)
        self.p = upd(self.p, IP)
        self.d = torch.stack([upd(self.d[:, i], ID + 3 * i) for i in range(2)], 1)
        self.bg = self.bg + delta[:, IBG:IBG + 3]
        self.ba = self.ba + delta[:, IBA:IBA + 3]
