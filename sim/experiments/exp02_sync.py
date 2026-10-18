"""Exp02: robustness to anchor sync jitter and receiver clock drift (reverberant room).
Run from repo root:  python -m sim.experiments.exp02_sync"""
import time

from sim.cipher.config import Scenario
from sim.experiments.common import crlb_avg, run, save, summarize

METHODS = ('CIPHER', 'TDOA', 'DRAPE-like', 'SCALA-AOA+TDOA')


def main(n_poses=24, n_frames=2):
    out = {'jitter': {}, 'drift': {}}
    for jit in [0.0, 2e-6, 10e-6, 50e-6, 200e-6]:
        scn = Scenario()
        scn.anchors.sync_jitter = jit
        t = time.time()
        poses, res = run(scn, n_poses, n_frames, seed=2, methods=METHODS)
        s = summarize(res, METHODS)
        s['CRLB_coh'] = dict(zip(('pos_rmse', 'psi_rmse'), crlb_avg(scn, poses, include=('coh',))))
        s['CRLB_coh+sync'] = dict(zip(('pos_rmse', 'psi_rmse'),
                                      crlb_avg(scn, poses, sigma_toa=max(jit, 1e-7))))
        out['jitter'][f'{jit * 1e6:g}us'] = s
        print(f'== sync jitter {jit * 1e6:g} us ({time.time() - t:.0f} s)')
        for m in METHODS + ('CRLB_coh', 'CRLB_coh+sync'):
            v = s[m]
            print(f"  {m:16s} pos {v['pos_rmse']:10.2f} mm" +
                  (f"  psi {v['psi_rmse']:9.4f} deg" if v.get('psi_rmse') is not None else ''))
    for ppm in [-100.0, 0.0, 100.0]:
        scn = Scenario()
        scn.clock.drift_ppm = ppm
        poses, res = run(scn, n_poses, n_frames, seed=3, methods=('CIPHER',))
        s = summarize(res, ('CIPHER',))
        out['drift'][f'{ppm:+g}ppm'] = s
        print(f"== receiver SFO {ppm:+g} ppm: CIPHER pos {s['CIPHER']['pos_rmse']:.2f} mm "
              f"psi {s['CIPHER']['psi_rmse']:.4f} deg")
    save('exp02_sync', out)


if __name__ == '__main__':
    main()
