"""Exp08: long runs (30 s) with blockage -- does CIPHER-X keep its fix, detect blocked anchors /
cycle slips, and recover? Model speed of sound off by 0.3 %, sync 2 us, sigma_s estimated online.
CIPHER alone and CIPHER-X (carrier-aided tracking) run on the same signals. Events (system time): 8-10 s anchor 1 direct path -25 dB (person in the line of sight, reflections
remain); 15-15.5 s anchor 3 silent; 22-23 s anchors 0 and 2 direct paths -25 dB (two at once).
Run from repo root:  python -m sim.experiments.exp08_long"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher import carrier
from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import track
from sim.cipher.pipeline import frame_starts
from sim.cipher.geometry import Lissajous, wrap
from sim.experiments.common import save
from sim.experiments.exp03_motion import Shifted
from sim.experiments.exp07_carrier import calib_frames, setup

EVENTS = [(1, 8.0, 10.0, 25.0, 'direct'), (3, 15.0, 15.5, 100.0, 'all'),
          (0, 22.0, 23.0, 25.0, 'direct'), (2, 22.0, 23.0, 25.0, 'direct')]
PHASES = [('clear 0-8 s', 0, 8), ('A1 blocked 8-10 s', 8, 10), ('clear 10-15 s', 10, 15),
          ('A3 silent 15-15.5 s', 15, 15.5), ('clear 15.5-22 s', 15.5, 22),
          ('A0+A2 blocked 22-23 s', 22, 23), ('clear 23-30 s', 23, 30)]


def run(args):
    seed, vmax, bias, n_frames = args
    scn = Scenario()
    scn.seed, scn.signal.scheme = seed, 'checker'
    scn.anchors.occlusions = list(EVENTS)
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
        rows.append([tr.sys_time(t_loc), base[k][0], base[k][1],
                     np.linalg.norm(r['p'] - p), abs(wrap(r['psi'] - psi)), float(r['fixed']),
                     r.get('n_anchors', 0), ct.sig_s])
    return rows


def main(n_seeds=4, n_frames=300, workers=4):
    fc = np.array([0.5 * (a + b) for a, b in Scenario().signal.bands])
    out = {}
    rm = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))) if len(x) else float('nan')
    with ProcessPoolExecutor(max_workers=workers) as ex:
        C = [r for rr in ex.map(calib_frames, [(s, 2e-6) for s in range(900, 904)]) for r in rr]
        bias, _ = carrier.calibrate_bias([c[0] for c in C], [c[1] for c in C], fc)
        for vmax in (0.5, 1.0):
            t = time.time()
            R = np.array(list(ex.map(run, [(s, vmax, bias, n_frames) for s in range(n_seeds)])))
            res = {}
            for name, a, b in PHASES:
                m = (R[..., 0] >= a) & (R[..., 0] < b)
                fx = m & (R[..., 5] > 0)
                res[name] = dict(cipher=1e3 * rm(R[..., 1][m]), x=1e3 * rm(R[..., 3][m]),
                                 x_psi=np.rad2deg(rm(R[..., 4][m])), cipher_psi=np.rad2deg(rm(R[..., 2][m])),
                                 fixed=float(fx.sum() / max(m.sum(), 1)),
                                 fixed_x=1e3 * rm(R[..., 3][fx]), max_x=1e3 * float(R[..., 3][m].max()),
                                 worse_than_cipher_3mm=float(np.mean(R[..., 3][m] > R[..., 1][m] + 3e-3)))
                v = res[name]
                print(f"v{vmax} {name:24s}: CIPHER {v['cipher']:6.2f} mm | CIPHER-X {v['x']:6.2f} mm (fixed {100 * v['fixed']:3.0f}%, "
                      f"on fixed {v['fixed_x']:.2f} mm, max {v['max_x']:.1f}) | worse than CIPHER by >3 mm: "
                      f"{100 * v['worse_than_cipher_3mm']:.0f}%", flush=True)
            res['sigma_s_final_us'] = (1e6 * R[:, -1, 7]).tolist()
            res['series'] = {'t': R[0, :, 0].tolist(), 'fixed_mean': R[..., 5].mean(0).tolist(),
                             'cipher_mm': (1e3 * R[..., 1]).tolist(), 'x_mm': (1e3 * R[..., 3]).tolist()}
            out[f'v{vmax}'] = res
            save('exp08_long', out)                                   # partial results survive interruptions
            print(f'v{vmax}: {time.time() - t:.0f} s', flush=True)
    save('exp08_long', out)


if __name__ == '__main__':
    main()
