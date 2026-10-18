"""Exp12: linearised checks of the CIPHER-X propositions (theory.md sec. 12) -- no signal
simulation, Fisher information / least squares on the measurement geometry only.
P1 crossover vs sync residual, P2 sensitivity to per-mic phase bias, P3 sound speed jointly,
P4 bootstrapped single/multi-frame integer success vs sound-speed prior, P5 FDMA alias.
Per-mic phase noise is set so that the array-only CRLB equals exp01's 2.0 mm.
Run from repo root:  python -m sim.experiments.exp12_theory_checks"""
import numpy as np
from scipy.stats import norm

from sim.cipher.config import Scenario
from sim.experiments.common import save

SCN = Scenario()
A = np.array(SCN.anchors.pos)
H = SCN.array.height
C = SCN.c
B = SCN.array.mic_body()
FC = np.array([0.5 * (a + b) for a, b in SCN.signal.bands])
I, M = len(A), len(B)
E_ANCH = np.kron(np.eye(I), np.ones((M, 1)))            # per-anchor common phase columns


def phases(z):
    """Per-(anchor, mic) carrier phase, z = [x, y, psi, t_rx, s]."""
    x, y, psi, trx, s = z
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    q = np.array([x, y, H]) + B @ Rz.T
    return 2 * np.pi * FC[:, None] * (np.linalg.norm(q[None] - A[:, None], axis=-1) / (C * (1 + s)) + trx)


def jac(z, e=1e-7):
    return np.array([((phases(z + d) - phases(z - d)) / (2 * e)).reshape(-1) for d in np.eye(5) * e]).T


def info(z, sig_phi, sig_s, tied, s_free):
    """Fisher matrix and Jacobian. tied: anchors' absolute phases tied to t_rx + d/c + eps_i."""
    J = jac(z)
    cols = [0, 1, 2] + ([3] if tied else []) + ([4] if s_free else [])
    nuis = E_ANCH * (np.repeat(2 * np.pi * FC, M)[:, None] if tied else 1.0)
    Js = np.hstack([J[:, cols], nuis])
    F = Js.T @ Js / sig_phi ** 2
    if tied:
        F[len(cols):, len(cols):] += np.eye(I) / sig_s ** 2
    return F, Js, len(cols)


def pos_crlb(z, sig_phi, sig_s, tied, s_free=False):
    Cm = np.linalg.inv(info(z, sig_phi, sig_s, tied, s_free)[0])
    return np.sqrt(Cm[0, 0] + Cm[1, 1])


def bias(z, sig_phi, sig_s, tied, dchi):
    _, Js, nc = info(z, sig_phi, sig_s, tied, False)
    rows, rhs = Js / sig_phi, dchi.reshape(-1) / sig_phi
    if tied:
        rows = np.vstack([rows, np.hstack([np.zeros((I, nc)), np.eye(I) / sig_s])])
        rhs = np.concatenate([rhs, np.zeros(I)])
    d = np.linalg.lstsq(rows, rhs, rcond=None)[0]
    return np.hypot(d[0], d[1])


def _carrier_rows(traj, sig_s, anchors, fc, with_n):
    K, Ii = len(traj), len(anchors)
    npar = 3 * K + 1 + (Ii if with_n else 0)
    rows = []
    for k, p in enumerate(traj):
        q = np.array([p[0], p[1], H])
        d = np.linalg.norm(anchors - q, axis=1)
        u = (q - anchors) / d[:, None]
        for i in range(Ii):
            r = np.zeros(npar)
            r[2 * k:2 * k + 2] = u[i, :2] / C
            r[2 * K + k] = 1.0
            r[3 * K] = -d[i] / C
            if with_n:
                r[3 * K + 1 + i] = 1 / fc[i]
            rows.append(r / sig_s)
    return np.array(rows), npar


def amb_success(traj, sig_s, sig_a, sig_sp, anchors=A, fc=FC):
    """Bootstrapped success of the cross-anchor integers over K frames (integers constant)."""
    K, Ii = len(traj), len(anchors)
    R, npar = _carrier_rows(traj, sig_s, anchors, fc, True)
    F = R.T @ R
    for k in range(K):
        F[2 * k, 2 * k] += 1 / sig_a ** 2
        F[2 * k + 1, 2 * k + 1] += 1 / sig_a ** 2
    F[3 * K, 3 * K] += 1 / sig_sp ** 2
    keep = [j for j in range(npar) if j != 3 * K + 1]   # gauge: first integer fixed
    Q = np.linalg.inv(F[np.ix_(keep, keep)])[-(Ii - 1):, -(Ii - 1):]
    return float(np.prod(2 * norm.cdf(1 / (2 * np.diag(np.linalg.cholesky(Q)))) - 1))


def alias_dchi2(traj, sig_s, sig_a, sig_sp):
    """Unabsorbed chi^2 of the common-shift alias pattern m_i = 1/f_i - mean(1/f)."""
    m = 1 / FC - np.mean(1 / FC)
    K = len(traj)
    R, npar = _carrier_rows(traj, sig_s, A, FC, False)
    b = np.tile(m, K) / sig_s
    pri = np.zeros((2 * K + 1, npar))
    for k in range(2 * K):
        pri[k, k] = 1 / sig_a
    pri[-1, 3 * K] = 1 / sig_sp
    Am, bm = np.vstack([R, pri]), np.concatenate([b, np.zeros(2 * K + 1)])
    res = bm - Am @ np.linalg.lstsq(Am, bm, rcond=None)[0]
    return float(res @ res)


def _line(rng, K, step=0.05):
    p0, ang = rng.uniform(2, 8, 2), rng.uniform(0, 2 * np.pi)
    return [p0 + step * k * np.array([np.cos(ang), np.sin(ang)]) for k in range(K)]


def main(n_poses=200):
    rng = np.random.default_rng(1)
    poses = [np.array([rng.uniform(1, 9), rng.uniform(1, 9), rng.uniform(-np.pi, np.pi), 0.0, 0.0])
             for _ in range(n_poses)]
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    sig_phi = 2.03e-3 / rm([pos_crlb(z, 1.0, 1e-6, False) for z in poses])
    out = {'sigma_phi_deg': float(np.rad2deg(sig_phi))}
    dch = [np.deg2rad(5.0) * rng.standard_normal((I, M)) for _ in poses]
    out['array_only'] = dict(crlb_mm=1e3 * rm([pos_crlb(z, sig_phi, 1e-6, False) for z in poses]),
                             bias5deg_mm=1e3 * rm([bias(z, sig_phi, 1e-6, False, d) for z, d in zip(poses, dch)]))
    print(f"P1/P2 array only: CRLB {out['array_only']['crlb_mm']:.2f} mm, "
          f"5 deg mic bias -> {out['array_only']['bias5deg_mm']:.2f} mm")
    out['tied'] = {}
    for ss in (0.5, 1, 2, 5, 10, 20, 50):
        v = dict(crlb_mm=1e3 * rm([pos_crlb(z, sig_phi, ss * 1e-6, True) for z in poses]),
                 crlb_s_free_mm=1e3 * rm([pos_crlb(z, sig_phi, ss * 1e-6, True, True) for z in poses]),
                 bias5deg_mm=1e3 * rm([bias(z, sig_phi, ss * 1e-6, True, d) for z, d in zip(poses, dch)]))
        out['tied'][f'{ss}us'] = v
        print(f"P1/P2/P3 tied, sync {ss:4g} us: CRLB {v['crlb_mm']:.2f} mm (s free {v['crlb_s_free_mm']:.2f}), "
              f"5 deg bias -> {v['bias5deg_mm']:.2f} mm")
    out['ambiguity'] = {}
    A6 = np.vstack([A, [[5.0, 0.2, 3.0], [5.0, 9.8, 2.95]]])
    f6 = np.concatenate([FC, [16250.0, 23750.0]])
    for lab, anc, fc in (('4 anchors', A, FC), ('6 anchors', A6, f6)):
        for s_sig in (3e-3, 1e-4):
            res = []
            for K in (1, 5, 10, 20):
                ps = [amb_success(_line(rng, K), 2e-6, 3e-3, s_sig, anc, fc) for _ in range(40)]
                out['ambiguity'][f'{lab}, s prior {s_sig:g}, K={K}'] = float(np.median(ps))
                res.append(float(np.median(ps)))
            print(f"P4 {lab}, s prior {100 * s_sig:.2f}%: success K=1/5/10/20 = " + '/'.join(f'{r:.3f}' for r in res))
    m = 1 / FC - np.mean(1 / FC)
    out['alias'] = {'m_us': (1e6 * m).tolist()}
    for K in (1, 10):
        res = []
        for ss in (1, 2, 4, 8):
            v = float(np.median([alias_dchi2(_line(rng, K), ss * 1e-6, 5e-3, 1e-4) for _ in range(60)]))
            out['alias'][f'K={K}, sigma_s={ss}us'] = v
            res.append(v)
        print(f"P5 alias |m| = {1e6 * np.linalg.norm(m):.2f} us, K={K}: delta-chi2 at sigma_s 1/2/4/8 us = " +
              '/'.join(f'{r:.1f}' for r in res))
    save('exp12_theory_checks', out)


if __name__ == '__main__':
    main()
