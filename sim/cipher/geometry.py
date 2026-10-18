"""Rigid-array geometry and trajectories."""
import numpy as np


def rotz(psi):
    """Rotation about z; psi scalar or (N,) -> (3,3) or (N,3,3)."""
    c, s = np.cos(psi), np.sin(psi)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def mic_world(p, psi, h, mic_body):
    """World mic positions. p (...,2), psi (...) -> (...,M,3)."""
    p = np.asarray(p, float)
    psi = np.asarray(psi, float)
    centre = np.concatenate([p, np.full(p.shape[:-1] + (1,), h)], -1)
    return centre[..., None, :] + np.einsum('...ij,mj->...mi', rotz(psi), mic_body)


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


class Static:
    def __init__(self, p, psi):
        self.p = np.asarray(p, float)
        self.psi = float(psi)

    def __call__(self, t):
        t = np.asarray(t, float)
        return np.broadcast_to(self.p, t.shape + (2,)), np.full(t.shape, self.psi)


class Lissajous:
    """Smooth closed trajectory inside the room; heading follows velocity plus a wobble."""

    def __init__(self, centre, amp, period, wobble_deg=20.0):
        self.c, self.a, self.T = np.asarray(centre, float), np.asarray(amp, float), float(period)
        self.wob = np.deg2rad(wobble_deg)

    def __call__(self, t):
        t = np.asarray(t, float)
        w = 2 * np.pi / self.T
        x = self.c[0] + self.a[0] * np.sin(w * t)
        y = self.c[1] + self.a[1] * np.sin(2 * w * t)
        vx, vy = self.a[0] * w * np.cos(w * t), 2 * self.a[1] * w * np.cos(2 * w * t)
        psi = np.unwrap(np.arctan2(vy, vx)) + self.wob * np.sin(3 * w * t) if t.ndim else \
            np.arctan2(vy, vx) + self.wob * np.sin(3 * w * t)
        return np.stack([x, y], -1), psi
