"""Exp05: aperture-ambiguity trade-off. Array radius sweep (reverberant room): accuracy, ambiguity
(gross heading error) rate, ratio-test margin, and CRLB.
Run from repo root:  python -m sim.experiments.exp05_aperture"""
import time

import numpy as np

from sim.cipher.config import Scenario
from sim.experiments.common import crlb_avg, run, save, summarize


def main(n_poses=24, n_frames=2):
    out = {}
    for R in [0.016, 0.025, 0.035, 0.050, 0.070, 0.100]:
        scn = Scenario()
        scn.array.radius = R
        t = time.time()
        poses, res = run(scn, n_poses, n_frames, seed=5, methods=('CIPHER',), workers=8)
        s = summarize(res, ('CIPHER',))['CIPHER']
        ratios = np.array([r for x in res for r in x['ratio']])
        s['ratio_med'], s['ratio_min'] = float(np.median(ratios)), float(ratios.min())
        s['crlb_pos'], s['crlb_psi'] = crlb_avg(scn, poses, include=('coh',))
        out[f'R{R*1e3:g}mm'] = s
        print(f"R {R*1e3:5.1f} mm: pos RMSE {s['pos_rmse']:7.2f} mm (med {s['pos_med']:.2f})  psi RMSE "
              f"{s['psi_rmse']:.4f} deg (med {s['psi_med']:.4f})  gross {100*s['gross']:.0f}%  ratio med "
              f"{s['ratio_med']:.2f} min {s['ratio_min']:.2f}  | CRLB {s['crlb_pos']:.2f} mm {s['crlb_psi']:.4f} deg"
              f"  ({time.time()-t:.0f} s)")
    save('exp05_aperture', out)


if __name__ == '__main__':
    main()
