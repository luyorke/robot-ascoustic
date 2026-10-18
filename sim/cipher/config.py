"""Scenario configuration. All units SI (m, s, Hz, Pa) unless noted."""
from dataclasses import dataclass, field

import numpy as np

C_SOUND = 343.0  # m/s at ~20 degC


@dataclass
class ArrayCfg:
    radius: float = 0.050          # docs/hardware.md: mic package centres on R = 50 mm
    n_mics: int = 8
    height: float = 0.50           # array plane height above floor
    # theta_k = 90 deg - 45 deg*(k-1), PDF top view (to be confirmed on hardware)
    first_angle_deg: float = 90.0
    step_deg: float = -45.0
    # mounting tilt of the array plane w.r.t. the robot body (roll about body x, pitch about body
    # y, degrees): unknown to the estimator unless calibrated (0.25 deg -> ~14 mm for CIPHER)
    tilt_deg: tuple = (0.0, 0.0)

    def mic_angles(self) -> np.ndarray:
        return np.deg2rad(self.first_angle_deg + self.step_deg * np.arange(self.n_mics))

    def mic_body(self) -> np.ndarray:
        """(M,3) nominal mic positions in body frame (array centre at origin, z=0, level)."""
        a = self.mic_angles()
        return np.stack([self.radius * np.cos(a), self.radius * np.sin(a), np.zeros_like(a)], 1)


def tilt_matrix(roll_deg, pitch_deg):
    """Rotation of the array plane: roll about body x, then pitch about body y."""
    r, p = np.deg2rad(roll_deg), np.deg2rad(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return Ry @ Rx


@dataclass
class SignalCfg:
    fs: float = 48_000.0
    chirp_len: float = 0.040       # T
    frame_len: float = 0.100       # T_f, all anchors emit concurrently (FDMA) once per frame
    bands: tuple = ((17_000, 18_500), (18_500, 20_000), (20_000, 21_500), (21_500, 23_000))
    taper: float = 0.1             # Tukey taper fraction (limits audible clicks)
    scheme: str = 'checker'        # 'up': all up-chirps; 'checker': anchors alternate up/down chirps
                                   # frame by frame, neighbours in opposite directions

    def direction(self, i, k):
        """+1 up-chirp / -1 down-chirp for anchor i in frame k."""
        if self.scheme == 'checker':
            return 1 if (i + k) % 2 == 0 else -1
        return 1


@dataclass
class RoomCfg:
    size: tuple = (10.0, 10.0, 3.5)
    reflection: float = 0.7        # pressure reflection coefficient (all surfaces); 0 -> free field
    max_order: int = 2             # image-source order
    air_db_per_m: float = 0.5      # air absorption near 20 kHz
    beta: tuple = None             # optional per-surface (x0, x1, y0, y1, floor, ceiling); overrides
    # --- effects the estimator does NOT model (robustness tests, exp10) ---
    scatterers: list = field(default_factory=list)  # point scatterers (x, y, z, g): furniture / clutter
    body_plate: tuple = None       # (depth below the array plane m, reflection coeff[, half-length,
                                   # half-width m in the body frame]): robot body / AGV deck

    def betas(self) -> np.ndarray:
        return np.asarray(self.beta if self.beta is not None else (self.reflection,) * 6, float)


@dataclass
class AnchorCfg:
    pos: np.ndarray = field(default_factory=lambda: np.array(
        [[0.2, 0.2, 3.00], [9.8, 0.2, 2.90], [9.8, 9.8, 3.10], [0.2, 9.8, 3.05]]))
    aim: np.ndarray = None         # (I,3) unit axes; default: towards room centre at array height
    spl_30cm: float = 90.0         # YZ3840: >= 90 dB SPL @30 cm, 20 kHz, 5 Vrms
    piston_radius: float = 0.01325  # YZ3840 mouth diameter 26.5 mm
    sync_jitter: float = 2e-6      # std of residual inter-anchor sync error per frame (s)
    # blockage events (anchor, t_start, t_end, attenuation dB, 'direct' | 'all'): 'direct' = a
    # person / object in the line of sight (reflections remain), 'all' = anchor silent / dropped
    occlusions: list = field(default_factory=list)

    def attenuation(self, i, t):
        """(direct-path dB, all-path dB) attenuation of anchor i for an emission at time t."""
        d = a = 0.0
        for j, t0, t1, db, kind in self.occlusions:
            if j == i and t0 <= t < t1:
                if kind == 'all':
                    a += db
                else:
                    d += db
        return d, a

    def axes(self, room: RoomCfg, h: float) -> np.ndarray:
        if self.aim is not None:
            return self.aim / np.linalg.norm(self.aim, axis=1, keepdims=True)
        centre = np.array([room.size[0] / 2, room.size[1] / 2, h])
        v = centre - self.pos
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@dataclass
class ClockCfg:
    offset: float = 0.0123         # receiver clock offset delta (s), unknown to the estimator
    drift_ppm: float = 30.0        # receiver SFO eta (ppm), unknown to the estimator


@dataclass
class MicCfg:
    sens_dbfs: float = -26.0       # dBFS @ 94 dB SPL
    snr_dba: float = 62.0          # self-noise 32 dB(A) SPL
    ambient_spl: float = 25.0      # in-band ambient acoustic noise (dB SPL, 17-23 kHz)
    f_res: float = 16_500.0        # Helmholtz resonance (datasheet peak ~16-17 kHz)
    q_res: float = 5.6             # ~ +15 dB peak
    f_res_spread: float = 0.0      # relative std of per-mic resonance frequency
    phase_err_deg: float = 0.0     # extra per-mic, frequency-flat phase error std
    pos_err: float = 0.0           # per-mic port position error std (m), in-plane
    lr_skew: float = 0.0           # PDM: mics with L/R=GND (U2,U4,U6,U8) sampled this much later (s),
                                   # half a PDM clock = 163 ns at 3.072 MHz


@dataclass
class Scenario:
    array: ArrayCfg = field(default_factory=ArrayCfg)
    signal: SignalCfg = field(default_factory=SignalCfg)
    room: RoomCfg = field(default_factory=RoomCfg)
    anchors: AnchorCfg = field(default_factory=AnchorCfg)
    clock: ClockCfg = field(default_factory=ClockCfg)
    mic: MicCfg = field(default_factory=MicCfg)
    c: float = C_SOUND
    seed: int = 0
    hw_seed: int = None            # if set, mic hardware errors use this seed (same "board" across runs)
    c_anchor_rel: tuple = None     # per-anchor relative speed-of-sound offsets (temperature non-uniformity)
