"""Exp09: complexity reductions for the embedded port -- lag-domain decimation q and number of
modelled paths (reverberant room, static, tracking-mode CIPHER). Python time is overhead-bound;
the decisive output is the accuracy at each reduction (FLOP estimate: docs/firmware_plan.md sec. 4).
Run from repo root:  python -m sim.experiments.exp09_complexity"""
import time

import numpy as np

from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import cipher
from sim.cipher.frontend import FrontEnd
from sim.cipher.geometry import Static, wrap
from sim.cipher.lagmodel import build_pulses
from sim.experiments.common import model_of, random_pose, save

CASES = [(1, 10), (4, 10), (8, 10), (8, 6), (8, 4), (8, 3)]


def main(n_poses=10):
    E = {c: [] for c in CASES}
    Tm = {c: [] for c in CASES}
    for seed in range(n_poses):
        scn = Scenario()
        scn.seed = seed
        p, psi = random_pose(np.random.default_rng(seed), scn)
        y, tr = synthesize(scn, Static(p, psi), 1)
        st = int(round(tr.delta * scn.signal.fs)) + 50
        mdl = model_of(scn)
        for q, L in CASES:
            fe = FrontEnd(scn, lag_decim=q)
            lag = (build_pulses(fe, scn), scn.room.size, True)
            obs = fe.process(y, st)
            est0 = cipher(mdl, obs, lag=lag, n_paths=L)                       # cold start
            t = time.time()
            est = cipher(mdl, obs, lag=lag, n_paths=L, prior=(est0.p, est0.psi))
            Tm[(q, L)].append(time.time() - t)
            E[(q, L)].append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
    out = {}
    for c in CASES:
        e = np.array(E[c])
        out[f'decim{c[0]}_paths{c[1]}'] = dict(pos_rmse_mm=1e3 * float(np.sqrt(np.mean(e[:, 0] ** 2))),
                                               psi_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(e[:, 1] ** 2)))),
                                               python_time_s=float(np.mean(Tm[c])))
        v = out[f'decim{c[0]}_paths{c[1]}']
        print(f"decim {c[0]}, paths {c[1]:2d}: pos {v['pos_rmse_mm']:.2f} mm  psi {v['psi_rmse_deg']:.4f} deg | "
              f"tracking-mode Python time {v['python_time_s']:.2f} s", flush=True)
    save('exp09_complexity', out)


if __name__ == '__main__':
    main()
