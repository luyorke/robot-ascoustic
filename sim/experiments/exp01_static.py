"""Exp01: static random poses, all methods vs CRLB, free field and reverberant room.
Run from repo root:  python -m sim.experiments.exp01_static"""
import time

from sim.cipher.config import Scenario
from sim.experiments.common import crlb_avg, crlb_lag_avg, run, save, summarize


def main(n_poses=32, n_frames=3):
    out = {}
    for name, refl in [('free_field', 0.0), ('reverb_r0.7_o2', 0.7)]:
        scn = Scenario()
        scn.room.reflection = refl
        t = time.time()
        poses, res = run(scn, n_poses, n_frames)
        s = summarize(res)
        s['CRLB_coh'] = dict(zip(('pos_rmse', 'psi_rmse'), crlb_avg(scn, poses, include=('coh',))))
        s['CRLB_coh+sync2us'] = dict(zip(('pos_rmse', 'psi_rmse'), crlb_avg(scn, poses, sigma_toa=2e-6)))
        s['CRLB_lag_mp+sync2us'] = dict(zip(('pos_rmse', 'psi_rmse'), crlb_lag_avg(scn, poses, sigma_sync=2e-6)))
        ratios = [r for x in res for r in x['ratio']]
        s['CIPHER']['ratio_min'] = float(min(ratios))
        out[name] = s
        print(f'== {name} ({time.time() - t:.0f} s, {n_poses} poses x {n_frames} frames)')
        for m, v in s.items():
            print(f"  {m:16s} pos RMSE {v['pos_rmse']:9.3f} mm"
                  + (f"   psi RMSE {v['psi_rmse']:8.4f} deg" if v.get('psi_rmse') is not None else '')
                  + (f"   med {v['pos_med']:.2f} mm / {v['psi_med']:.4f} deg  gross(>1deg) {100*v['gross']:.0f}%"
                     if v.get('psi_med') is not None else ''))
    save('exp01_static', out)


if __name__ == '__main__':
    main()
