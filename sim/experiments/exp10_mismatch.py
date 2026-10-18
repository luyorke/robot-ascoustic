"""Exp10: robustness to effects the estimator does not model ("inverse crime" check).
Cases: point-scatterer clutter (furniture), a reflector moving with the robot just below the array
(robot body / top plate; extra path of a few cm, unresolvable), and per-anchor speed-of-sound
offsets (temperature non-uniformity between the anchors' paths). Moving robot 0.5 m/s, checker
waveform, model speed of sound off by 0.3 %; carrier constants calibrated in the same environment.
Run from repo root:  python -m sim.experiments.exp10_mismatch"""
import copy
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher import carrier
from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import cipher_frame, track
from sim.cipher.pipeline import frame_starts
from sim.cipher.geometry import Lissajous, Static, wrap
from sim.experiments.common import random_pose, save
from sim.experiments.exp03_motion import Shifted
from sim.experiments.exp07_carrier import setup

K_PER_DEGC = 1.7e-3                     # relative change of c per degC near 20 degC


def clutter(n, g, seed=11):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0.5, 9.5), rng.uniform(0.5, 9.5), rng.uniform(0.2, 1.5), g) for _ in range(n)]


CASES = {
    'baseline': {},
    'clutter 6 x -20 dB': dict(scatterers=clutter(6, 0.10)),
    'clutter 6 x -12 dB': dict(scatterers=clutter(6, 0.25)),
    'body plate 10 cm, r=0.3': dict(body_plate=(0.10, 0.3)),
    'body plate, modelled': dict(body_plate=(0.10, 0.3), model_plate=0.10),   # depth from the robot CAD
    'temperature 0.1 degC': dict(dT=0.1),
    'temperature 0.3 degC': dict(dT=0.3),
}


def scenario(case, seed):
    scn = Scenario()
    scn.seed, scn.signal.scheme = seed, 'checker'
    cfg = CASES[case]
    if 'scatterers' in cfg:
        scn.room.scatterers = copy.deepcopy(cfg['scatterers'])
    if 'body_plate' in cfg:
        scn.room.body_plate = cfg['body_plate']
    if 'dT' in cfg:                     # fixed per room (same in calibration and test)
        scn.c_anchor_rel = tuple(K_PER_DEGC * cfg['dT'] * np.random.default_rng(77).standard_normal(4))
    return scn


def calib(args):
    case, seed = args
    scn = scenario(case, seed)
    p, psi = random_pose(np.random.default_rng(seed), scn)
    y, tr = synthesize(scn, Static(p, psi), 2)
    fe, mdl, lag = setup(scn)
    mdl.body_plate = CASES[case].get('model_plate')
    out = []
    for k in range(2):
        st = int(round((k * scn.signal.frame_len * (1 + tr.eta) + tr.delta) * scn.signal.fs)) + 50
        est, _ = cipher_frame(mdl, fe, y, st, passes=1, lag=lag)
        out.append((est, p))
    return out


def run(args):
    case, seed, bias, n_frames, vmax = args
    scn = scenario(case, seed)
    w = vmax / 6.7
    rng = np.random.default_rng(seed)
    traj = Shifted(Lissajous((5.0, 5.0), (3.0, 3.0), 2 * np.pi / w, 20.0), rng.uniform(0, 2 * np.pi / w), 0.0)
    y, tr = synthesize(scn, traj, n_frames + 1)
    fe, mdl, lag = setup(scn)
    mdl.body_plate = CASES[case].get('model_plate')
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
                     abs(wrap(r['psi'] - psi)), float(r['fixed']), ct.sig_s])
    return rows


def main(n_seeds=8, n_frames=24, vmax=0.5, workers=4):
    fc = np.array([0.5 * (a + b) for a, b in Scenario().signal.bands])
    rm = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))) if len(x) else float('nan')
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for case in CASES:
            t = time.time()
            C = [r for rr in ex.map(calib, [(case, s) for s in range(900, 904)]) for r in rr]
            bias, _ = carrier.calibrate_bias([c[0] for c in C], [c[1] for c in C], fc)
            R = np.array(list(ex.map(run, [(case, s, bias, n_frames, vmax) for s in range(n_seeds)])))
            fx = R[..., 4] > 0
            v = dict(cipher=1e3 * rm(R[..., 0]), cipher_psi=np.rad2deg(rm(R[..., 1])), x=1e3 * rm(R[..., 2]),
                     x_psi=np.rad2deg(rm(R[..., 3])), fixed=float(fx.mean()), fixed_x=1e3 * rm(R[..., 2][fx]),
                     fixed_cipher=1e3 * rm(R[..., 0][fx]),
                     worse_fix=float(np.mean(R[..., 2][fx] > R[..., 0][fx] + 3e-3)) if fx.any() else 0.0,
                     sigma_s_us=float(np.median(1e6 * R[:, -1, 5])))
            out[case] = v
            save('exp10_mismatch', out)
            print(f"{case:26s}: CIPHER {v['cipher']:6.2f} mm / {v['cipher_psi']:.4f} deg | CIPHER-X {v['x']:6.2f} mm / "
                  f"{v['x_psi']:.4f} deg, fixed {100 * v['fixed']:.0f}% (on fixed {v['fixed_x']:.2f} vs {v['fixed_cipher']:.2f}), "
                  f"bad fixes {100 * v['worse_fix']:.0f}%, sigma_s est {v['sigma_s_us']:.1f} us ({time.time() - t:.0f} s)",
                  flush=True)
    save('exp10_mismatch', out)


if __name__ == '__main__':
    main()
