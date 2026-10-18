"""Anchor waveforms: FDMA linear up-chirps with a Tukey taper."""
import numpy as np


def tukey_cont(u, alpha):
    """Continuous Tukey window on u in [0,1]; zero outside."""
    u = np.asarray(u, float)
    w = np.zeros_like(u)
    inside = (u >= 0) & (u <= 1)
    w[inside] = 1.0
    if alpha > 0:
        a2 = alpha / 2
        lo = inside & (u < a2)
        hi = inside & (u > 1 - a2)
        w[lo] = 0.5 * (1 - np.cos(np.pi * u[lo] / a2))
        w[hi] = 0.5 * (1 - np.cos(np.pi * (1 - u[hi]) / a2))
    return w


def chirp_at(t, f0, bw, T, taper, direction=1):
    """Real chirp evaluated at arbitrary times t (s) since emission start; exact (no interpolation).
    direction +1 sweeps f0 -> f0+bw, -1 sweeps f0+bw -> f0 (same band)."""
    t = np.asarray(t, float)
    mu = bw / T
    if direction < 0:
        return tukey_cont(t / T, taper) * np.cos(2 * np.pi * ((f0 + bw) * t - 0.5 * mu * t * t))
    return tukey_cont(t / T, taper) * np.cos(2 * np.pi * (f0 * t + 0.5 * mu * t * t))


def reference(sig, i, direction=1):
    """Sampled nominal chirp of anchor i (what the receiver correlates against)."""
    f0, f1 = sig.bands[i]
    t = np.arange(int(round(sig.chirp_len * sig.fs))) / sig.fs
    return chirp_at(t, f0, f1 - f0, sig.chirp_len, sig.taper, direction)

