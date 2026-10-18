"""Exp11: AGV deck below the array -- the strongest unmodelled reflector (exp10). Its depth is not
given by the user: it is estimated in the motion-capture calibration (lagmodel.plate_depth_scan)
and then modelled as one more image path. Deck 0.7 x 0.5 m, metal (r = 0.8), at 5 / 12 cm below
the array; moving robot 0.5 m/s, model speed of sound off by 0.3 %.
Run from repo root:  python -m sim.experiments.exp11_agv"""
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
from sim.cipher.lagmodel import build_pulses, plate_depth_scan
from sim.experiments.common import model_of, random_pose, save
from sim.experiments.exp03_motion import Shifted
from sim.experiments.exp07_carrier import setup

DECKS = {'no deck': None, 'deck 5 cm': (0.05, 0.8, 0.35, 0.25), 'deck 12 cm': (0.12, 0.8, 0.35, 0.25)}


def scenario(deck, seed):
    scn = Scenario()
    scn.seed, scn.signal.scheme = seed, 'checker'
    scn.room.body_plate = DECKS[deck]
    return scn


def calib(args):
    deck, seed, depth = args
    scn = scenario(deck, seed)
    p, psi = random_pose(np.random.default_rng(seed), scn)
    y, tr = synthesize(scn, Static(p, psi), 2)
    fe, mdl, lag = setup(scn)
    mdl.body_plate = depth
    out = []
    for k in range(2):
        st = int(round((k * scn.signal.frame_len * (1 + tr.eta) + tr.delta) * scn.signal.fs)) + 50
        if depth == 'raw':                                     # observations for the depth scan
            out.append((fe.process(y, st), p, psi))
            continue
        est, _ = cipher_frame(mdl, fe, y, st, passes=1, lag=lag)
        out.append((est, p))
    return out


def run(args):
    deck, depth, seed, bias, n_frames, vmax = args
    scn = scenario(deck, seed)
    w = vmax / 6.7
    rng = np.random.default_rng(seed)
    traj = Shifted(Lissajous((5.0, 5.0), (3.0, 3.0), 2 * np.pi / w, 20.0), rng.uniform(0, 2 * np.pi / w), 0.0)
    y, tr = synthesize(scn, traj, n_frames + 1)
    fe, mdl, lag = setup(scn)
    mdl.body_plate = depth
    fs, T = scn.signal.fs, scn.signal.frame_len
    fc = np.array([0.5 * (a + b) for a, b in scn.signal.bands])
    starts = frame_starts(fe, y, n_frames=n_frames)                  # no receiver-clock truth used
    base = []
    for est, t_loc in track(mdl, fe, y, starts, lag):
        p, psi = tr.pose(t_loc)
        base.append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
    ct = carrier.CarrierTracker(fc, bias, T, sigma_s=4e-6, window=10, diff_min=8.0)
    rows = []
    for k, (est, t_loc) in enumerate(track(mdl, fe, y, starts, lag, aid=ct.update)):
        p, psi = tr.pose(t_loc)
        r = est.extra['aided']
        rows.append([base[k][0], base[k][1], np.linalg.norm(r['p'] - p), abs(wrap(r['psi'] - psi)), float(r['fixed'])])
    return rows


def main(decks=None, n_seeds=8, n_frames=24, vmax=0.5, workers=4):
    import json
    import os
    fc = np.array([0.5 * (a + b) for a, b in Scenario().signal.bands])
    rm = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))) if len(x) else float('nan')
    path = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp11_agv.json')
    out = json.load(open(path)) if decks and os.path.exists(path) else {}   # merge partial reruns
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for deck in (decks or DECKS):
            raw = [r for rr in ex.map(calib, [(deck, s, 'raw') for s in range(900, 904)]) for r in rr]
            scn = Scenario()
            fe = FrontEnd(scn)
            depth_est, _ = plate_depth_scan(model_of(scn), raw, build_pulses(fe, scn), scn.room.size)
            for label, depth in (('unmodelled', None), ('calibrated depth', depth_est)):
                if deck == 'no deck' and label == 'calibrated depth' and depth_est is None:
                    continue
                t = time.time()
                C = [r for rr in ex.map(calib, [(deck, s, depth) for s in range(900, 904)]) for r in rr]
                bias, _ = carrier.calibrate_bias([c[0] for c in C], [c[1] for c in C], fc)
                R = np.array(list(ex.map(run, [(deck, depth, s, bias, n_frames, vmax) for s in range(n_seeds)])))
                fx = R[..., 4] > 0
                key = f'{deck} / {label}'
                v = dict(true_depth=None if DECKS[deck] is None else DECKS[deck][0],
                         est_depth=None if depth is None else float(depth),
                         cipher=1e3 * rm(R[..., 0]), cipher_psi=np.rad2deg(rm(R[..., 1])),
                         x=1e3 * rm(R[..., 2]), x_psi=np.rad2deg(rm(R[..., 3])), fixed=float(fx.mean()),
                         fixed_x=1e3 * rm(R[..., 2][fx]), fixed_cipher=1e3 * rm(R[..., 0][fx]),
                         worse_fix=float(np.mean(R[..., 2][fx] > R[..., 0][fx] + 3e-3)) if fx.any() else 0.0)
                out[key] = v
                save('exp11_agv', out)
                print(f"{key:30s} (depth est {v['est_depth']}): CIPHER {v['cipher']:6.2f} mm / {v['cipher_psi']:.4f} deg | "
                      f"CIPHER-X {v['x']:6.2f} mm, fixed {100 * v['fixed']:.0f}% (on fixed {v['fixed_x']:.2f} vs "
                      f"{v['fixed_cipher']:.2f}), bad fixes {100 * v['worse_fix']:.0f}% ({time.time() - t:.0f} s)", flush=True)
    save('exp11_agv', out)


if __name__ == '__main__':
    import sys
    main(decks=sys.argv[1:] or None)       # e.g.  python -m sim.experiments.exp11_agv "deck 12 cm"
