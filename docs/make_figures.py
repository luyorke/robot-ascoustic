"""Figures for docs/CIPHER理论与算法框架.docx (data from sim/results/*.json).
Run from repo root:  python docs/make_figures.py   ->  docs/figures/fig*.png"""
import json
import os
import sys

import matplotlib
import matplotlib.patches
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from sim.cipher.config import Scenario                             # noqa: E402

RES = os.path.join(ROOT, 'sim', 'results')
OUT = os.path.join(ROOT, 'docs', 'figures')
os.makedirs(OUT, exist_ok=True)

# reference palette (dataviz skill, light mode): fixed slot order, never cycled
BLUE, ORANGE, AQUA, YELLOW, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
X_COL, C_COL, BASE_COL = BLUE, ORANGE, MUTED                       # CIPHER-X, CIPHER, other baselines

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'lines.linewidth': 2.0, 'figure.dpi': 110, 'savefig.dpi': 220, 'axes.axisbelow': True,
})


def load(name):
    with open(os.path.join(RES, name + '.json'), encoding='utf-8') as f:
        return json.load(f)


def save(fig, name):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')
    for ax in fig.axes:
        if ax.get_yscale() == 'log':
            ax.yaxis.set_major_formatter(fmt)
        if ax.get_xscale() == 'log':
            ax.xaxis.set_major_formatter(fmt)
        if any(isinstance(c, matplotlib.patches.Rectangle) and c.get_width() < 1.5 for c in ax.patches[:3])                 and ax.get_xscale() == 'linear' and not ax.lines[:1]:
            ax.grid(axis='x', visible=False)                      # vertical bars: no vertical grid
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', name)


def bar_labels(ax, bars, fmt='{:.1f}'):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8, color=INK2)


# ---------------------------------------------------------------- 1 system
def fig_system():
    scn = Scenario()
    A = np.array(scn.anchors.pos)
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes([0.03, 0.08, 0.40, 0.86])
    ax.add_patch(Rectangle((0, 0), 10, 10, fill=False, ec=INK2, lw=1.5))
    for i, a in enumerate(A):
        ax.add_patch(Polygon([[a[0] - .25, a[1] - .25], [a[0] + .25, a[1] - .25], [a[0], a[1] + .3]], color=ORANGE))
        ax.text(a[0] + (0.35 if a[0] < 5 else -0.35), a[1] + (0.35 if a[1] < 5 else -0.45),
                f'基站 {i}\n{scn.signal.bands[i][0]/1e3:.1f}–{scn.signal.bands[i][1]/1e3:.1f} kHz\n高 {a[2]:.2f} m',
                ha='left' if a[0] < 5 else 'right', va='bottom' if a[1] < 5 else 'top', fontsize=8, color=INK,
                bbox=dict(fc='white', ec='none', pad=1.5))
    p = np.array([4.2, 5.6])
    ax.add_patch(FancyBboxPatch(p - [0.45, 0.32], 0.9, 0.64, boxstyle='round,pad=0.02', fc='#dbe8f8', ec=BLUE, lw=1.5))
    ax.add_patch(Circle(p, 0.18, fill=False, ec=BLUE, lw=1.5))
    ax.text(p[0], p[1] - 0.62, 'AGV + 8 麦克风圆阵\n（异步接收）', ha='center', va='top', fontsize=8.5, color=INK,
            bbox=dict(fc='white', ec='none', pad=1.5))
    for a in A:
        ax.plot([a[0], p[0]], [a[1], p[1]], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.annotate('', xy=(9.8, -0.75), xytext=(0.2, -0.75), arrowprops=dict(arrowstyle='<->', color=AQUA, lw=1.4))
    ax.text(5, -0.62, '基站间无线同步（μs 级残差 σ_s）', ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.1, 10.6)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('(a) 场景：10 × 10 × 3.5 m 房间顶视图', loc='left')
    ax.grid(False)
    # array
    ax2 = fig.add_axes([0.49, 0.12, 0.22, 0.78])
    th = np.deg2rad(90 - 45 * np.arange(8))
    ax2.add_patch(Circle((0, 0), 56.9, fc='#f3f3f1', ec=INK2, lw=1))
    ax2.add_patch(Circle((0, 0), 45.0, fc='white', ec=INK2, lw=1))
    for k, t in enumerate(th):
        ax2.add_patch(Circle((50 * np.cos(t), 50 * np.sin(t)), 3.0, color=BLUE))
        ax2.text(62 * np.cos(t), 62 * np.sin(t), f'U{k+1}', ha='center', va='center', fontsize=8, color=INK)
    ax2.annotate('', xy=(50, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=INK2))
    ax2.text(25, 4, 'R = 50 mm', ha='center', fontsize=8, color=INK2)
    ax2.text(0, -80, '相邻间距 38 mm ≈ 2λ（20 kHz）\n→ 严重栅瓣', ha='center', fontsize=8.5, color=INK)
    ax2.set_xlim(-75, 75)
    ax2.set_ylim(-95, 75)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(b) 阵列 WB-SD2008（PDM MEMS ×8）', loc='left', fontsize=11)
    # side view
    ax3 = fig.add_axes([0.76, 0.12, 0.23, 0.78])
    ax3.plot([0, 10], [0, 0], color=INK2, lw=2)
    ax3.plot([0, 10], [3.5, 3.5], color=INK2, lw=1)
    ax3.add_patch(Polygon([[0.25, 3.18], [0.25, 2.82], [0.62, 3.0]], color=ORANGE))
    ax3.add_patch(Rectangle((5.6, 0.12), 1.4, 0.26, fc='#dbe8f8', ec=BLUE))
    ax3.plot([6.3], [0.5], 'o', color=BLUE, ms=6)
    ax3.plot([0.5, 6.3], [3.0, 0.5], color=MUTED, lw=1, ls=(0, (4, 3)))
    ax3.plot([0.5, 4.6, 6.3], [3.0, 0.0, 0.5], color=AQUA, lw=1)
    ax3.plot([0.5, 6.0, 6.3], [3.0, 0.38, 0.5], color=VIOLET, lw=1)
    ax3.text(2.3, 0.15, '地面反射', fontsize=8, color=INK)
    ax3.text(6.6, 0.95, 'AGV 车体\n反射', fontsize=8, color=INK)
    ax3.text(0.2, 3.2, '基站 ~3 m', fontsize=8, color=INK)
    ax3.text(6.5, 0.5, '阵列 0.5 m', fontsize=8, color=INK, va='center')
    ax3.set_xlim(0, 9.5)
    ax3.set_ylim(-0.3, 3.8)
    ax3.axis('off')
    ax3.set_title('(c) 侧视：主要多径', loc='left', fontsize=11)
    save(fig, 'fig01_system.png')


# ---------------------------------------------------------------- 2 waveform
def fig_waveform():
    sig = Scenario().signal
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    for k in range(3):
        t0 = k * sig.frame_len * 1e3
        ax.axvspan(t0, t0 + sig.chirp_len * 1e3, color='#f3f3f1', lw=0, zorder=0)
        ax.text(t0 + 2, 23.25, f'第 {k} 帧', fontsize=8, color=INK2)
        for i, (f0, f1) in enumerate(sig.bands):
            d = sig.direction(i, k)
            fa, fb = (f0, f1) if d > 0 else (f1, f0)
            ax.plot([t0, t0 + sig.chirp_len * 1e3], [fa / 1e3, fb / 1e3], color=cols[i], lw=2.4,
                    label=f'基站 {i}' if k == 0 else None, solid_capstyle='round')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('频率 (kHz)')
    ax.set_xlim(-5, 300)
    ax.set_ylim(16.8, 23.6)
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=9)
    ax.set_title('棋盘式上/下扫 FDMA 波形：基站 i 第 k 帧扫频方向 (−1)^(i+k)，40 ms chirp / 100 ms 帧', loc='left', pad=26)
    save(fig, 'fig02_waveform.png')


# ---------------------------------------------------------------- 3 framework
def box(ax, xy, w, h, text, fc, ec, fs=8.6, bold_first=True):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.012,rounding_size=0.015', fc=fc, ec=ec, lw=1.2))
    lines = text.split('\n')
    y = xy[1] + h - 0.028
    for j, ln in enumerate(lines):
        ax.text(xy[0] + 0.012, y, ln, fontsize=fs + (0.6 if j == 0 else 0), va='top', ha='left', color=INK,
                fontweight='bold' if (j == 0 and bold_first) else 'normal')
        y -= 0.031 if j == 0 else 0.027


def arrow(ax, a, b, col=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, color=col, lw=1.2))


def fig_framework():
    fig = plt.figure(figsize=(11, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.10, 0.98)
    gray, blue, orange, green = '#f3f3f1', '#e3eefb', '#fdebe2', '#e1f4ec'
    box(ax, (0.02, 0.82), 0.20, 0.14, '① 8 通道录音\nPDM → 48 kHz PCM\n异步：钟差 δ、钟漂 η 未知\n帧周期 T_f = 100 ms', gray, MUTED)
    box(ax, (0.26, 0.82), 0.22, 0.14, '② 前端\n4 子带 × 上/下扫参考脉压\n首径检测（8 ms 回溯）\n扫频方向自动判别', gray, MUTED)
    box(ax, (0.52, 0.82), 0.22, 0.14, '③ 自主帧定时\n捕获：最早到达前 5 ms\n跟踪：一阶环路\n无需任何接收钟信息', gray, MUTED)
    box(ax, (0.78, 0.82), 0.20, 0.14, '④ 时延窗基带信号\n直达径附近 −2…+4 ms\n可 8× 降采样（无损）', gray, MUTED)
    arrow(ax, (0.225, 0.89), (0.255, 0.89))
    arrow(ax, (0.485, 0.89), (0.515, 0.89))
    arrow(ax, (0.745, 0.89), (0.775, 0.89))
    # CIPHER
    ax.add_patch(FancyBboxPatch((0.02, 0.36), 0.46, 0.40, boxstyle='round,pad=0.01', fc='white', ec=ORANGE, lw=1.6))
    ax.text(0.03, 0.745, 'CIPHER：时钟免疫的阵元间载波相位（每帧）', fontsize=10.5, fontweight='bold', color=INK, va='top')
    box(ax, (0.035, 0.585), 0.195, 0.125, 'S1 门控\nTDOA（两帧上/下扫联立，\n抗多普勒）+ 留一法', orange, ORANGE)
    box(ax, (0.27, 0.585), 0.195, 0.125, 'S2 模糊消解\n位姿域相干度 Λ(p, ψ)\n航向全搜索 + ratio test', orange, ORANGE)
    box(ax, (0.035, 0.375), 0.43, 0.185,
        'S3 时延域几何感知多径模型（变量投影 + LM）\n直达 + 房间镜像径 + AGV 车体反射；每基站 t₀、每路径复增益\n'
        '参数：x, y, ψ, t₀,i, t_rx, v, ω, η, 声速 s\n距离–多普勒耦合 κβ、波前采样像差 1/(1+β)、按路径时间压缩脉冲\n'
        '跟踪：帧内 v/ω 作先验中心；失锁（稳健相干度/跳变）→ 冷启动', orange, ORANGE, fs=8.2)
    arrow(ax, (0.235, 0.648), (0.265, 0.648))
    arrow(ax, (0.36, 0.585), (0.36, 0.56))
    # CIPHER-X
    ax.add_patch(FancyBboxPatch((0.52, 0.36), 0.46, 0.40, boxstyle='round,pad=0.01', fc='white', ec=BLUE, lw=1.6))
    ax.text(0.53, 0.745, 'CIPHER-X：跨基站载波相位（多帧）', fontsize=10.5, fontweight='bold', color=INK, va='top')
    box(ax, (0.535, 0.585), 0.195, 0.125, 'X1 载波观测\n直达径增益在脉冲峰处的相位\n→ chirp 中心到达（含 β² 修正）', blue, BLUE)
    box(ax, (0.77, 0.585), 0.195, 0.125, 'X2 状态\n声速 s（CIPHER 喂入）、σ_s 自估\n窗口钟漂 η、帧间整周传递', blue, BLUE)
    box(ax, (0.535, 0.375), 0.43, 0.185,
        'X3 捕获 → 保持（GNSS 式 float → fix → hold）\n10 帧窗口批量最小二乘；所有整周候选共享投影矩阵；差值检验\n'
        '逐基站整周、残差 4σ 剔除、恢复后取整重固定\n完好性：CIPHER 独立检验、窗口平均新息、σ_s ≤ 4 μs\n'
        '固定解反馈为 CIPHER 下一帧先验（载波辅助跟踪）', blue, BLUE, fs=8.2)
    arrow(ax, (0.735, 0.648), (0.765, 0.648))
    arrow(ax, (0.86, 0.585), (0.86, 0.56))
    ax.plot([0.88, 0.88, 0.25], [0.82, 0.79, 0.79], color=INK2, lw=1.2)
    arrow(ax, (0.25, 0.79), (0.25, 0.765))
    arrow(ax, (0.75, 0.79), (0.75, 0.765))
    arrow(ax, (0.475, 0.50), (0.525, 0.50), col=ORANGE)
    arrow(ax, (0.525, 0.43), (0.475, 0.43), col=BLUE)
    # calibration
    box(ax, (0.02, 0.12), 0.46, 0.19,
        '标定（动捕真值，一次）\n• 阵元：每阵元声孔偏移 + 每（子带, 阵元）复增益 / 群时延（≥50 帧）\n'
        '• AGV 车体反射深度：候选深度扫描，取残差最小（改善 < 1 dB 判为无）\n• 每基站载波常数 b_i：与标定时声速联合，圆周平均\n'
        '• 阵元编号手性：相干度判别', green, AQUA, fs=8.4)
    box(ax, (0.52, 0.12), 0.46, 0.19,
        '输出\n• 位置：固定时 ~1 mm（CIPHER-X），否则 ~4 mm（CIPHER）\n• 航向：~0.02°（阵列）\n'
        '• 速度、角速度、接收钟漂、声速（≈0.1 °C 声学温度计）\n• 完好性标志：固定/浮点/失锁', green, AQUA, fs=8.4)
    arrow(ax, (0.25, 0.31), (0.25, 0.36), col=AQUA)
    arrow(ax, (0.75, 0.36), (0.75, 0.31), col=AQUA)
    save(fig, 'fig03_framework.png')


# ---------------------------------------------------------------- 4 hierarchy (schematic)
def fig_hierarchy():
    x = np.linspace(-60, 60, 4000)                                 # mm along a line through the truth
    lam = 343 / 20e3 * 1e3
    env = np.exp(-0.5 * (x / 25.0) ** 2)
    arr = 0.86 * np.exp(-0.5 * ((x - 0) / 4.0) ** 2) + 0.14 * np.exp(-0.5 * (x / 30.0) ** 2) \
        + 0.70 * np.exp(-0.5 * ((x - 38) / 4.0) ** 2) + 0.66 * np.exp(-0.5 * ((x + 41) / 4.0) ** 2)
    car = 0.5 + 0.5 * np.cos(2 * np.pi * x / lam)
    fig, axs = plt.subplots(3, 1, figsize=(8.6, 5.6), sharex=True)
    for ax, y, col, title in zip(axs, [env, arr, car], [MUTED, ORANGE, BLUE],
                                 ['① 包络 TDOA（带宽 1.5 kHz）：单峰、宽，误差 cm 级 → 门控',
                                  '② 阵元间载波相位（R = 50 mm）：位姿域栅瓣，主瓣宽 mm 级 → CIPHER，误差 ~4 mm',
                                  f'③ 跨基站载波相位：周期 λ ≈ {lam:.0f} mm 的梳状，需整周固定 → CIPHER-X，误差 ~1 mm']):
        ax.plot(x, y, color=col)
        ax.fill_between(x, 0, y, color=col, alpha=0.12, lw=0)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_title(title, loc='left', fontsize=9.6)
        ax.axvline(0, color=INK, lw=0.8, ls=(0, (2, 2)))
    axs[1].axvspan(-25, 25, color=MUTED, alpha=0.12, lw=0)
    axs[2].axvspan(-6, 6, color=ORANGE, alpha=0.15, lw=0)
    axs[1].text(-24, 0.95, '①的门限', fontsize=8, color=INK2, va='top')
    axs[2].text(-5.5, 1.05, '②的门限', fontsize=8, color=INK2, va='top')
    axs[2].set_xlabel('沿某方向的位置偏差 (mm)（示意，尺度按本系统参数）')
    fig.tight_layout()
    save(fig, 'fig04_hierarchy.png')


# ---------------------------------------------------------------- 5 P1/P2
def fig_crossover():
    d = load('exp12_theory_checks')
    ss = [0.5, 1, 2, 5, 10, 20, 50]
    crlb = [d['tied'][f'{s:g}us']['crlb_mm'] for s in ss]
    crlb_s = [d['tied'][f'{s:g}us']['crlb_s_free_mm'] for s in ss]
    bias = [d['tied'][f'{s:g}us']['bias5deg_mm'] for s in ss]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    ax.plot(ss, crlb, 'o-', color=X_COL, ms=5, label='跨基站载波（声速已知）')
    ax.plot(ss, crlb_s, 's--', color=AQUA, ms=5, label='跨基站载波（声速联合估计）')
    ax.axhline(d['array_only']['crlb_mm'], color=C_COL, lw=2, label='仅阵元间相位（CIPHER）')
    ax.axvline(6, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.text(6.4, 0.25, 'σ*≈6 μs', fontsize=8.5, color=INK2)
    ax.set_xscale('log')
    ax.set_xlabel('基站同步残差 σ_s (μs)')
    ax.set_ylabel('位置 CRLB (mm)')
    ax.set_title('(a) 命题 1：交叉点', loc='left')
    ax.legend(fontsize=8.5, loc='upper left')
    ax = axs[1]
    ax.plot(ss, bias, 'o-', color=X_COL, ms=5, label='跨基站载波')
    ax.axhline(d['array_only']['bias5deg_mm'], color=C_COL, lw=2, label='仅阵元间相位')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('基站同步残差 σ_s (μs)')
    ax.set_ylabel('位置偏差 (mm)')
    ax.set_title('(b) 命题 2：每阵元 5° 相位误差的影响', loc='left')
    ax.legend(fontsize=8.5, loc='lower right')
    fig.tight_layout()
    save(fig, 'fig05_crossover.png')


# ---------------------------------------------------------------- 6 P4 / P5
def fig_ambiguity():
    d = load('exp12_theory_checks')
    Ks = [1, 5, 10, 20]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    for lab, col, ls in (('4 anchors', X_COL, '-'), ('6 anchors', AQUA, '--')):
        for sp, mk, name in ((0.003, 'o', '±0.3%'), (0.0001, 's', '±0.01%')):
            y = [d['ambiguity'][f'{lab}, s prior {sp:g}, K={K}'] for K in Ks]
            ax.plot(Ks, y, marker=mk, ls=ls, color=col if sp < 1e-3 else C_COL, ms=5,
                    label=f'{"4" if lab.startswith("4") else "6"} 个基站，声速先验 {name}')
    ax.set_xticks(Ks)
    ax.set_xlabel('累积帧数 K')
    ax.set_ylabel('整周固定成功率（bootstrapping）')
    ax.set_ylim(0.6, 1.02)
    ax.set_title('(a) 命题 4：成败由声速精度主导', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    ax = axs[1]
    ss = [1, 2, 4, 8]
    for K, col in ((1, C_COL), (10, X_COL)):
        ax.plot(ss, [d['alias'][f'K={K}, sigma_s={s}us'] for s in ss], 'o-', color=col, ms=5, label=f'K = {K} 帧')
    ax.axhline(8, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.05, 9.5, '检验门限 γ = 8', fontsize=8.5, color=INK2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(ss)
    ax.set_xticklabels([str(s) for s in ss])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('基站同步残差 σ_s (μs)')
    ax.set_ylabel('公共平移别名的可检测量 Δχ²')
    m = d['alias']['m_us']
    ax.set_title(f'(b) 命题 5：FDMA 宽巷别名（‖m‖ = {np.linalg.norm(m):.1f} μs）', loc='left')
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, 'fig06_ambiguity.png')


# ---------------------------------------------------------------- 7 exp01 + exp05
def fig_static():
    d = load('exp01_static')
    r = d['reverb_r0.7_o2']
    names = [('CIPHER', 'CIPHER'), ('TDOA', 'TDOA'), ('DRAPE-like', 'DRAPE 式'), ('SCALA-AOA+TDOA', 'SCALA 式')]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    vals = [r[k]['pos_rmse'] for k, _ in names]
    cols = [C_COL] + [BASE_COL] * 3
    bars = ax.bar([n for _, n in names], vals, color=cols, width=0.6)
    bar_labels(ax, bars)
    crl = r['CRLB_lag_mp+sync2us']
    crl = crl['pos_rmse'] if isinstance(crl, dict) else crl
    ax.axhline(crl, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(3.35, crl * 1.15, f'CRLB {crl:.1f} mm', fontsize=8, color=INK, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('位置 RMSE (mm，对数)')
    ax.set_title('(a) 静止单帧（混响）：与现有方法对比', loc='left')
    for i, (k, _) in enumerate(names):
        g = r[k].get('gross')
        if g:
            ax.text(i, vals[i] * 0.55, f'{100*g:.0f}% 粗差', ha='center', fontsize=7.8, color='white')
    ax = axs[1]
    e = load('exp05_aperture')
    R = [16, 25, 35, 50, 70, 100]
    ax.plot(R, [e[f'R{x}mm']['pos_rmse'] for x in R], 'o-', color=C_COL, ms=5, label='CIPHER（仿真）')
    cr = [e[f'R{x}mm'].get('crlb_mm', np.nan) for x in R]
    if np.all(np.isfinite(cr)):
        ax.plot(R, cr, 's--', color=INK2, ms=4, label='CRLB')
    ax.axvline(50, color=X_COL, lw=1, ls=(0, (3, 3)))
    ax.text(51, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] < 20 else 9, '本阵列 R = 50 mm', fontsize=8.5, color=X_COL)
    ax.set_xlabel('阵列半径 R (mm)')
    ax.set_ylabel('位置 RMSE (mm)')
    ax.set_title('(b) 口径–模糊折中（exp05）', loc='left')
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, 'fig07_static_aperture.png')


# ---------------------------------------------------------------- 8 exp02 / exp03 / exp06
def fig_robust_cipher():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    d = load('exp02_sync')
    js = ['0us', '2us', '10us', '50us', '200us']
    ax = axs[0]
    ax.plot([0.5, 2, 10, 50, 200], [d['jitter'][j]['CIPHER']['pos_rmse'] for j in js], 'o-', color=C_COL, ms=5, label='CIPHER')
    ax.plot([0.5, 2, 10, 50, 200], [d['jitter'][j]['TDOA']['pos_rmse'] for j in js], 's--', color=BASE_COL, ms=5, label='TDOA')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.5, 2, 10, 50, 200])
    ax.set_xticklabels(['0', '2', '10', '50', '200'])
    ax.set_xlabel('同步抖动 (μs)')
    ax.set_ylabel('位置 RMSE (mm)')
    ax.set_title('(a) 同步抖动（钟漂 ±100 ppm 无影响）', loc='left', fontsize=10)
    ax.legend(fontsize=8.5)
    ax = axs[1]
    m = load('exp03_motion')
    vs = ['0.5', '1.0', '2.0']
    w = 0.38
    xx = np.arange(3)
    b1 = ax.bar(xx - w / 2, [m[f'vmax{v}/checker, moving start']['CIPHER-track']['pos_rmse'] for v in vs], w, color=C_COL, label='运动模型 + 棋盘波形')
    b2 = ax.bar(xx + w / 2, [m[f'vmax{v}/checker, moving start']['CIPHER-no-motion-model']['pos_rmse'] for v in vs], w, color=BASE_COL, label='不建模运动')
    bar_labels(ax, b1)
    bar_labels(ax, b2, '{:.0f}')
    ax.set_yscale('log')
    ax.set_xticks(xx)
    ax.set_xticklabels([f'{v} m/s' for v in vs])
    ax.set_ylabel('位置 RMSE (mm，对数)')
    ax.set_title('(b) 运动中起步（exp03）', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[2]
    c = load('exp06_calibration')
    keys = [('uncalibrated', '未标定'), ('cal 12 frames', '12 帧'), ('cal 24 frames', '24 帧'), ('cal 48 frames', '48 帧'),
            ('oracle gains + ports (floor)', '模型下限')]
    bars = ax.bar([n for _, n in keys], [c[k]['pos_rmse'] for k, _ in keys],
                  color=[BASE_COL, C_COL, C_COL, C_COL, INK2], width=0.6)
    bar_labels(ax, bars)
    ax.set_ylabel('位置 RMSE (mm)')
    ax.set_title('(c) 阵元失配的动捕标定（exp06）', loc='left', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig08_cipher_robustness.png')


# ---------------------------------------------------------------- 9 exp07
def fig_carrier():
    d = load('exp07_carrier')
    keys = ['v0.5_sync2us', 'v1.0_sync2us', 'v2.0_sync2us']
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    xx = np.arange(3)
    w = 0.36
    b1 = ax.bar(xx - w / 2, [d[k]['fixed_cipher_pos'] for k in keys], w, color=C_COL, label='CIPHER（同一批帧）')
    b2 = ax.bar(xx + w / 2, [d[k]['fixed_x_pos'] for k in keys], w, color=X_COL, label='CIPHER-X（整周固定帧）')
    bar_labels(ax, b1, '{:.2f}')
    bar_labels(ax, b2, '{:.2f}')
    ax.set_xticks(xx)
    ax.set_xticklabels(['0.5 m/s', '1.0 m/s', '2.0 m/s'])
    ax.set_ylabel('位置 RMSE (mm)')
    ax.set_title('运动场景：模型声速偏 0.3%、帧定时自主获取、σ_s 在线自估（exp07）', loc='left', fontsize=10)
    ax.legend(fontsize=8.5)
    for i, k in enumerate(keys):
        ax.text(i, -0.12, f'固定率 {100*d[k]["fixed"]:.0f}%', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, 'fig09_carrier.png')


# ---------------------------------------------------------------- 10 exp08 time series
def fig_long():
    d = load('exp08_long')['v0.5']['series']
    t = np.array(d['t'])
    c = np.median(np.array(d['cipher_mm']), 0)
    x = np.median(np.array(d['x_mm']), 0)
    f = np.array(d['fixed_mean'])
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax = axs[0]
    for a, b, lab in ((8, 10, '基站 1 直达径被挡'), (15, 15.5, '基站 3 静默'), (22, 23, '基站 0+2 同时被挡')):
        for axx in axs:
            axx.axvspan(a, b, color=YELLOW, alpha=0.18, lw=0)
        ax.text((a + b) / 2, 400, lab, ha='center', fontsize=8, color=INK)
    ax.plot(t, c, color=C_COL, lw=1.5, label='CIPHER（4 条轨迹中位数）')
    ax.plot(t, x, color=X_COL, lw=1.5, label='CIPHER-X')
    ax.set_yscale('log')
    ax.set_ylim(0.2, 900)
    ax.set_ylabel('位置误差 (mm，对数)')
    ax.set_title('30 s 长轨迹（0.5 m/s）中的遮挡与恢复（exp08）', loc='left')
    ax.legend(fontsize=8.5, loc='upper left')
    ax = axs[1]
    ax.fill_between(t, 0, f, color=X_COL, alpha=0.35, lw=0, step='mid')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('固定比例')
    ax.set_xlabel('时间 (s)')
    fig.tight_layout()
    save(fig, 'fig10_long.png')


# ---------------------------------------------------------------- 11 exp10 / exp11
def fig_mismatch():
    d = load('exp10_mismatch')
    a = load('exp11_agv')
    rows = [('无失配', d['baseline']), ('家具散射 −20 dB', d['clutter 6 x -20 dB']),
            ('路径温差 0.1 °C', d['temperature 0.1 degC']),
            ('AGV 车体 5 cm，未建模', a['deck 5 cm / unmodelled']), ('AGV 车体 5 cm，标定深度', a['deck 5 cm / calibrated depth']),
            ('AGV 车体 12 cm，未建模', a['deck 12 cm / unmodelled']), ('AGV 车体 12 cm，标定深度', a['deck 12 cm / calibrated depth'])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(rows))[::-1]
    h = 0.38
    b1 = ax.barh(y + h / 2, [r['cipher'] for _, r in rows], h, color=C_COL, label='CIPHER（全部帧）')
    b2 = ax.barh(y - h / 2, [r['fixed_x'] for _, r in rows], h, color=X_COL, label='CIPHER-X（固定帧）')
    for b in list(b1) + list(b2):
        wv = b.get_width()
        if np.isfinite(wv):
            ax.text(wv * 1.08, b.get_y() + b.get_height() / 2, f'{wv:.1f}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows])
    ax.set_xscale('log')
    ax.set_xlabel('位置 RMSE (mm，对数)')
    ax.set_title('估计器未建模的效应（exp10）与 AGV 车体反射的标定建模（exp11）', loc='left', fontsize=10)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, 'fig11_mismatch.png')


if __name__ == '__main__':
    for fn in (fig_system, fig_waveform, fig_framework, fig_hierarchy, fig_crossover, fig_ambiguity,
               fig_static, fig_robust_cipher, fig_carrier, fig_long, fig_mismatch):
        fn()
