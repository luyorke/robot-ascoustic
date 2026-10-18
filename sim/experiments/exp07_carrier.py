"""Exp07: CIPHER-X -- cross-anchor carrier phase on top of CIPHER tracking (docs/review.md).
Moving robot (checker waveform, start while moving), model speed of sound off by 0.3 % (~0.5 degC),
per-anchor carrier constants calibrated from 8 static motion-capture frames; sync residual
estimated online by the tracker (initialised at 4 us). Last config: poor sync (10 us) -> the
tracker must stay in the CIPHER regime instead of fixing wrong integers.
Run from repo root:  python -m sim.experiments.exp07_carrier"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher import carrier
from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import cipher_frame, track
from sim.cipher.frontend import FrontEnd
from sim.cipher.geometry import Lissajous, Static, wrap
from sim.cipher.pipeline import frame_starts
from sim.cipher.lagmodel import build_pulses
from sim.experiments.common import model_of, random_pose, save
from sim.experiments.exp03_motion import Shifted

C_ERR = 0.003


def setup(scn):
    fe, mdl = FrontEnd(scn), model_of(scn)
    mdl.c = scn.c * (1 + C_ERR)
    return fe, mdl, (build_pulses(fe, scn), scn.room.size, True)


def calib_frames(args):
    """Two static frames at a known pose (motion capture)."""
    seed, sync = args
    scn = Scenario()
    scn.seed, scn.anchors.sync_jitter = seed, sync
    p, psi = random_pose(np.random.default_rng(seed), scn)
    y, tr = synthesize(scn, Static(p, psi), 2)
    fe, mdl, lag = setup(scn)
    out = []
    for k in range(2):
        st = int(round((k * scn.signal.frame_len * (1 + tr.eta) + tr.delta) * scn.signal.fs)) + 50
        est, _ = cipher_frame(mdl, fe, y, st, passes=1, lag=lag)
        out.append((est, p))
    return out


def run(args):
    seed, vmax, sync, bias, n_frames = args
    scn = Scenario()
    scn.seed, scn.signal.scheme, scn.anchors.sync_jitter = seed, 'checker', sync
    w = vmax / 6.7
    rng = np.random.default_rng(seed)
    traj = Shifted(Lissajous((5.0, 5.0), (3.0, 3.0), 2 * np.pi / w, 20.0), rng.uniform(0, 2 * np.pi / w), 0.0)
    y, tr = synthesize(scn, traj, n_frames + 1)
    fe, mdl, lag = setup(scn)
    fs, T = scn.signal.fs, scn.signal.frame_len
    fc = np.array([0.5 * (a + b) for a, b in scn.signal.bands])
    starts = frame_starts(fe, y, n_frames=n_frames)                  # no receiver-clock truth used
    base = []                                                            # CIPHER alone
    for est, t_loc in track(mdl, fe, y, starts, lag):
        p, psi = tr.pose(t_loc)
        base.append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
    ct = carrier.CarrierTracker(fc, bias, T, sigma_s=4e-6, window=10, diff_min=8.0)
    rows = []
    for k, (est, t_loc) in enumerate(track(mdl, fe, y, starts, lag, aid=ct.update)):   # CIPHER-X
        p, psi = tr.pose(t_loc)
        r = est.extra['aided']
        rows.append([base[k][0], base[k][1], np.linalg.norm(r['p'] - p),
                     abs(wrap(r['psi'] - psi)), float(r['fixed']), ct.s - (1 / (1 + C_ERR) - 1), ct.sig_s])
    return rows


def main(n_seeds=8, n_frames=20, workers=4):
    fc = np.array([0.5 * (a + b) for a, b in Scenario().signal.bands])
    out = {}
    rm = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for vmax, sync in [(0.5, 2e-6), (1.0, 2e-6), (2.0, 2e-6), (0.5, 10e-6)]:
            t = time.time()
            C = [r for rr in ex.map(calib_frames, [(s, sync) for s in range(900, 904)]) for r in rr]
            bias, s_cal = carrier.calibrate_bias([c[0] for c in C], [c[1] for c in C], fc)
            R = np.array([r for rr in ex.map(run, [(s, vmax, sync, bias, n_frames) for s in range(n_seeds)])
                          for r in rr]).reshape(n_seeds, n_frames, -1)
            fx = R[..., 4] > 0
            s = dict(cipher_pos=1e3 * rm(R[..., 0]), cipher_psi=np.rad2deg(rm(R[..., 1])),
                     x_pos=1e3 * rm(R[..., 2]), x_pos_med=1e3 * float(np.median(R[..., 2])),
                     x_psi=np.rad2deg(rm(R[..., 3])), fixed=float(fx.mean()),
                     fixed_x_pos=1e3 * rm(R[..., 2][fx]) if fx.any() else float('nan'),
                     fixed_cipher_pos=1e3 * rm(R[..., 0][fx]) if fx.any() else float('nan'),
                     wrong_fix=float(np.mean(R[..., 2][fx] > R[..., 0][fx] + 3e-3)) if fx.any() else 0.0,
                     fix_rate_by_frame=fx.mean(0).tolist(),
                     s_err_final_pct=100 * float(np.mean(np.abs(R[:, -1, 5]))),
                     sigma_s_final_us=(1e6 * R[:, -1, 6]).tolist(), sound_speed_cal_pct=100 * s_cal)
            key = f'v{vmax}_sync{sync * 1e6:g}us'
            out[key] = s
            save('exp07_carrier', out)                                   # partial results survive interruptions
            print(f"{key}: CIPHER {s['cipher_pos']:.2f} mm / {s['cipher_psi']:.4f} deg | CIPHER-X {s['x_pos']:.2f} mm "
                  f"(med {s['x_pos_med']:.2f}) / {s['x_psi']:.4f} deg | fixed {100 * s['fixed']:.0f}% -> on fixed frames "
                  f"{s['fixed_x_pos']:.2f} vs {s['fixed_cipher_pos']:.2f} mm, worse-than-CIPHER fixes {100 * s['wrong_fix']:.0f}% | "
                  f"sigma_s est {np.median(s['sigma_s_final_us']):.1f} us ({time.time() - t:.0f} s)", flush=True)
    save('exp07_carrier', out)


if __name__ == '__main__':
    main()
