"""Builds docs/CIPHER理论与算法框架.docx (figures from docs/make_figures.py, numbers from sim/results).
Run from repo root:  python docs/make_figures.py && python docs/build_report.py"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from docx import Document                                     # noqa: E402
from docx.enum.section import WD_ORIENT                       # noqa: E402,F401
from docx.enum.table import WD_TABLE_ALIGNMENT                # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK       # noqa: E402
from docx.oxml import OxmlElement                             # noqa: E402
from docx.oxml.ns import qn                                   # noqa: E402
from docx.shared import Cm, Pt, RGBColor                      # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIG = os.path.join(ROOT, 'docs', 'figures')
EQ = os.path.join(FIG, 'eq')
os.makedirs(EQ, exist_ok=True)
RES = os.path.join(ROOT, 'sim', 'results')
OUT = os.path.join(ROOT, 'docs', 'CIPHER理论与算法框架.docx')

BODY_CN, BODY_EN, HEAD_CN = 'SimSun', 'Times New Roman', 'Microsoft YaHei'
INK = RGBColor(0x0b, 0x0b, 0x0b)
ACCENT = RGBColor(0x2a, 0x78, 0xd6)


def R(name):
    with open(os.path.join(RES, name + '.json'), encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------- helpers
def set_font(run, cn=BODY_CN, en=BODY_EN, size=None, bold=None, color=None, italic=None):
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.append(rf)
    rf.set(qn('w:eastAsia'), cn)
    rf.set(qn('w:ascii'), en)
    rf.set(qn('w:hAnsi'), en)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def style_doc(doc):
    st = doc.styles['Normal']
    st.font.name = BODY_EN
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_CN)
    pf = st.paragraph_format
    pf.space_after = Pt(4)
    pf.line_spacing = 1.3
    for lvl, size in ((1, 16), (2, 13.5), (3, 11.5)):
        h = doc.styles[f'Heading {lvl}']
        h.font.name = HEAD_CN
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = INK if lvl > 1 else ACCENT
        rpr = h.element.get_or_add_rPr()
        rf = rpr.find(qn('w:rFonts'))
        if rf is None:
            rf = OxmlElement('w:rFonts')
            rpr.append(rf)
        rf.set(qn('w:eastAsia'), HEAD_CN)
        rf.set(qn('w:ascii'), HEAD_CN)
        rf.set(qn('w:hAnsi'), HEAD_CN)
        h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True


def H(doc, text, lvl):
    return doc.add_heading(text, level=lvl)


def P(doc, text, bold_lead=None, size=None, align=None, indent=True):
    """Paragraph; **...** segments are bold."""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align:
        p.alignment = align
    if bold_lead:
        r = p.add_run(bold_lead)
        set_font(r, bold=True, size=size)
    parts = text.split('**')
    for k, seg in enumerate(parts):
        if seg:
            r = p.add_run(seg)
            set_font(r, bold=(k % 2 == 1), size=size)
    return p


def B(doc, text, level=0):
    """Bullet with **bold** support."""
    p = doc.add_paragraph(style='List Bullet' if level == 0 else 'List Bullet 2')
    parts = text.split('**')
    for k, seg in enumerate(parts):
        if seg:
            r = p.add_run(seg)
            set_font(r, bold=(k % 2 == 1))
    p.paragraph_format.space_after = Pt(2)
    return p


_fig_no = [0]
_tab_no = [0]


def FIGR(doc, fname, caption, width_cm=16.0):
    _fig_no[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIG, fname), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f'图 {_fig_no[0]}  ')
    set_font(r, cn=HEAD_CN, en=HEAD_CN, size=9, bold=True)
    r = c.add_run(caption)
    set_font(r, size=9)
    c.paragraph_format.space_after = Pt(10)
    return _fig_no[0]


_eq_no = [0]


def EQN(doc, tex, name, fs=15):
    """Display equation rendered with matplotlib mathtext, numbered on the right."""
    _eq_no[0] += 1
    path = os.path.join(EQ, name + '.png')
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f'${tex}$', fontsize=fs)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04, transparent=False, facecolor='white')
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    width_cm = min(w / 300 * 2.54, 13.5)
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    widths = (Cm(14.5), Cm(1.5))
    for i, cw in enumerate(widths):
        tbl.columns[i].width = cw
        tbl.rows[0].cells[i].width = cw
    c0 = tbl.rows[0].cells[0].paragraphs[0]
    c0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c0.add_run().add_picture(path, width=Cm(width_cm))
    c1 = tbl.rows[0].cells[1].paragraphs[0]
    c1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = c1.add_run(f'({_eq_no[0]})')
    set_font(r, size=10)
    tbl.rows[0].cells[1].vertical_alignment = 1
    _no_borders(tbl)
    return _eq_no[0]


def _no_borders(tbl):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement('w:tblBorders')
    for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{e}')
        el.set(qn('w:val'), 'nil')
        b.append(el)
    tblPr.append(b)


def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear')
    sh.set(qn('w:color'), 'auto')
    sh.set(qn('w:fill'), hex_fill)
    tcPr.append(sh)


def TABLE(doc, caption, header, rows, widths_cm, font_size=9):
    _tab_no[0] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f'表 {_tab_no[0]}  ')
    set_font(r, cn=HEAD_CN, en=HEAD_CN, size=9, bold=True)
    r = c.add_run(caption)
    set_font(r, size=9)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths_cm):
        t.columns[j].width = Cm(w)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.width = Cm(widths_cm[j])
            cell.text = ''
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            parts = str(val).split('**')
            for k, seg in enumerate(parts):
                if seg:
                    rr = p.add_run(seg)
                    set_font(rr, size=font_size, bold=(i == 0 or k % 2 == 1),
                             cn=HEAD_CN if i == 0 else BODY_CN, en=HEAD_CN if i == 0 else BODY_EN)
            if i == 0:
                _shade(cell, 'E8EFF8')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return _tab_no[0]


def add_toc(doc):
    p = doc.add_paragraph()
    r = p.add_run()
    fld = OxmlElement('w:fldChar')
    fld.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = 'TOC \\o "1-2" \\h \\z \\u'
    sep = OxmlElement('w:fldChar')
    sep.set(qn('w:fldCharType'), 'separate')
    txt = OxmlElement('w:t')
    txt.text = '（打开文档时 Word 会提示更新域，选择"是"即生成目录；也可右键 → 更新域）'
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    for e in (fld, instr, sep, txt, end):
        r._r.append(e)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for kind, text in (('begin', None), (None, 'PAGE'), ('end', None)):
        if kind:
            e = OxmlElement('w:fldChar')
            e.set(qn('w:fldCharType'), kind)
        else:
            e = OxmlElement('w:instrText')
            e.set(qn('xml:space'), 'preserve')
            e.text = text
        r._r.append(e)
    set_font(r, size=9)


# ---------------------------------------------------------------- content
def build():
    e1, e2, e3, e5 = R('exp01_static'), R('exp02_sync'), R('exp03_motion'), R('exp05_aperture')
    e6, e7, e8, e9 = R('exp06_calibration'), R('exp07_carrier'), R('exp08_long'), R('exp09_complexity')
    e10, e11, e12 = R('exp10_mismatch'), R('exp11_agv'), R('exp12_theory_checks')
    rev = e1['reverb_r0.7_o2']
    crl = rev['CRLB_lag_mp+sync2us']
    crl = crl['pos_rmse'] if isinstance(crl, dict) else crl

    doc = Document()
    style_doc(doc)
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.0)
    page_numbers(sec)

    # ---- title page
    for _ in range(5):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('CIPHER / CIPHER-X')
    set_font(r, cn=HEAD_CN, en=HEAD_CN, size=28, bold=True, color=ACCENT)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('异步稀疏圆阵的声学载波相位定位定姿')
    set_font(r, cn=HEAD_CN, en=HEAD_CN, size=20, bold=True)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('理论贡献与算法框架')
    set_font(r, cn=HEAD_CN, en=HEAD_CN, size=16)
    for _ in range(3):
        doc.add_paragraph()
    for line in ('8 麦克风圆阵 WB-SD2008 + 4 个同步近超声基站（YZ3840）+ AGV 地面机器人',
                 '版本：v0.4（仿真验证完成，待实测）    日期：2026-10-17',
                 '代码与数据：sim/（Python 仿真）、sim/results/（实验结果）、firmware/（基站波形）'):
        t = doc.add_paragraph()
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = t.add_run(line)
        set_font(r, size=10.5, color=RGBColor(0x52, 0x51, 0x4e))
    page_break(doc)

    # ---- abstract + toc
    H(doc, '摘要', 1)
    P(doc, '本项目面向地面移动机器人（AGV）的室内高精度定位与定姿：4 个相互同步的近超声基站在 17–23 kHz 以频分方式发射 chirp，'
           'AGV 上的 8 麦克风圆阵（半径 50 mm）接收，**阵列与基站之间不同步**。现有声学定位工作只使用包络（群时延）信息，'
           '精度受 1–3 kHz 带宽限制在厘米级；本阵列阵元间距约 2 个波长，直接做相位测向会产生严重栅瓣。')
    P(doc, '我们提出两级载波相位方法。第一级 **CIPHER** 只使用"同一基站、不同阵元"之间的载波相位：它与接收钟差、钟漂、基站同步误差、'
           '基站硬件相位和平移多普勒完全无关；稀疏阵列的栅瓣通过多基站位姿域耦合与 TDOA 门控消除；时延域几何感知多径模型同时估计'
           '直达径、房间镜像径与 AGV 车体反射，并把运动（距离–多普勒耦合、波前采样像差）和声速作为参数。第二级 **CIPHER-X** 基于一个'
           '关键观察：**声学载波周期为 50 μs，μs 级无线同步的基站本身就已经是"载波相位同步"的**，因此跨基站的绝对载波相位可以像 GNSS '
           '载波相位定位一样使用，基线从阵列口径（0.1 m）扩大到房间尺度（约 10 m）；阵列负责航向与整周模糊度固定，声速由多径模型单帧估计。')
    P(doc, f'理论上给出 5 个命题：同步误差的交叉点（本系统约 6 μs）、对阵元误差敏感度降低约 ρ/R 倍、声速可观性、"整周可固定性由声速精度主导"，'
           f'以及 FDMA 宽巷别名决定的同步上限（约 4 μs）。仿真中（10×10×3.5 m 混响房间、模型声速偏 0.3%、帧定时自主获取），CIPHER 静止精度 '
           f'{rev["CIPHER"]["pos_rmse"]:.1f} mm / {rev["CIPHER"]["psi_rmse"]:.3f}°（现有方法 54–237 mm），CIPHER-X 在 0.5/1/2 m/s 运动中整周固定帧精度 '
           f'{e7["v0.5_sync2us"]["fixed_x_pos"]:.2f}/{e7["v1.0_sync2us"]["fixed_x_pos"]:.2f}/{e7["v2.0_sync2us"]["fixed_x_pos"]:.2f} mm，'
           '无错误固定；单基站遮挡下维持约 1 mm，信号恢复后 2–3 s 内重新固定。')
    H(doc, '目录', 1)
    add_toc(doc)
    page_break(doc)

    # ---- 1 background
    H(doc, '1  研究背景与问题', 1)
    H(doc, '1.1  硬件与场景', 2)
    FIGR(doc, 'fig01_system.png', '系统组成：(a) 房间与基站布局；(b) 8 麦克风圆阵；(c) 侧视图中的主要多径（地面、AGV 车体）。')
    B(doc, '**接收阵列**：WB-SD2008-V1.1，8 颗 LinkMems LMD3526T261 PDM MEMS 麦克风，均匀圆阵 R = 50 mm，主控 T113，采样率 48 kHz，安装在 AGV 上，离地约 0.5 m。')
    B(doc, '**基站**：4 个 YZ3840 压电超声喇叭（谐振 20 kHz，口径 26.5 mm），装在 10 × 10 × 3.5 m 房间四角、高约 3 m；基站间通过主站计数值广播实现 μs 级同步；波形可任意设置，帧周期 100 ms。')
    B(doc, '**真值**：动作捕捉 + 激光雷达；不使用 IMU。')
    H(doc, '1.2  核心难点', 2)
    B(doc, '**异步**：阵列与基站不同步，接收钟差 δ 与钟漂 η（数十 ppm）未知；基站之间也只有 μs 级同步残差 σ_s。')
    B(doc, '**稀疏阵列栅瓣**：20 kHz 时相邻阵元间距 38 mm ≈ 2λ，单基站相位测向的最高旁瓣仅比主瓣低 0.2–2.8 dB，无法直接使用。')
    B(doc, '**强多径**：墙角基站下地面反射只比直达径弱 3–6 dB、程差 0.3–0.5 m，小于单基站带宽分辨率 c/B ≈ 0.23 m 的数倍；AGV 车体反射程差仅几厘米。')
    B(doc, '**运动**：上扫 chirp 的距离–多普勒耦合把包络时延放大 f_c/B ≈ 13 倍；运动阵列在不同时刻采样同一波前。')
    B(doc, '**硬件与环境**：麦克风 16.5 kHz 谐振附近通道间幅相失配；声速随温度变化（0.5 °C ≈ 0.3%）。')
    H(doc, '1.3  现有工作的局限', 2)
    TABLE(doc, '代表性声学定位工作（papers/ 与检索文献）', ['工作', '同步方式', '观测量', '精度'],
          [['SCALA / A-LOAF', '基站异步', '小口径（R=2.5 cm）MUSIC 测向 + 群时延', '2.2 cm / 0.27°；11.7 cm'],
           ['LUNA', '异步', '相干加权 PHAT 角谱', '8.9 cm / 0.56°'],
           ['DRAPE', '基站同步', '逐阵元群时延 TDOA + 刚体配准', '3.6 cm / 2.35°'],
           ['HARP / PALM', '基站同步', '群时延 + 多斜率 chirp 消多普勒', '3.9 cm；5.8 cm'],
           ['CAT / MilliSonic', '需参考点', '单链路 FMCW 相位跟踪（相对量）', '数 mm（相对、短距）']],
          [3.0, 2.6, 6.2, 4.0])
    P(doc, '共同点：多基站 + 阵列的系统只用了**包络（群时延）信息**，精度受带宽 B 限制；载波相位只在单链路相对跟踪中出现过。'
           '"同步基站 + 异步接收机"在结构上与 GNSS 完全同构（卫星同步、接收机钟差与钟漂未知、多普勒、多天线测姿），但 GNSS 的载波相位与整周模糊度方法尚未被系统地迁移到声学阵列中。')

    # ---- 2 idea
    H(doc, '2  总体思路：分级载波相位', 1)
    P(doc, '全文只有一条主线：**声速慢，μs 级同步的声学基站就是载波相位同步的；异步稀疏阵列因此可以像 GNSS 一样做分级载波相位定位**。'
           '三级观测量的精度依次提高、模糊度依次加重，每一级为下一级提供门限（图 2）。')
    FIGR(doc, 'fig04_hierarchy.png', '三级观测量（示意，尺度按本系统参数）：包络 TDOA 单峰但宽；阵元间载波相位精细但有位姿域栅瓣；跨基站载波相位最精细，是周期约 17 mm 的梳状，需整周固定。', 14.5)
    TABLE(doc, '三级观测量的分工', ['层级', '观测量', '对钟/同步的要求', '精度', '模糊度', '本文处理'],
          [['①', '跨基站包络 TDOA', '消去接收钟差', 'cm（多径偏差）', '无', '门控（两帧上/下扫联立）'],
           ['②', '同一基站、跨阵元载波相位', '完全免疫', '~4 mm / 0.02°', '位姿域栅瓣', 'CIPHER：多基站耦合 + ratio test'],
           ['③', '跨基站绝对载波相位', 'σ_s ≲ 4 μs', '~1 mm', '每基站整周', 'CIPHER-X：GNSS 式 float→fix→hold']],
          [1.0, 3.6, 2.8, 2.6, 2.2, 3.8])

    # ---- 3 theory
    H(doc, '3  理论贡献', 1)
    P(doc, '下面每项贡献按"问题 → 结论 → 意义 → 证据"组织。命题 1–5 的数值核对由 sim/experiments/exp12_theory_checks.py 复现（线性化 Fisher 信息与最小二乘），'
           '其余结论由信号级仿真（exp01–exp11）验证。')

    H(doc, '3.1  C1：时钟免疫的阵元间载波相位与位姿域模糊消解（CIPHER）', 2)
    P(doc, '对基站 i、阵元 m，取同一基站跨阵元的互谱：')
    EQN(doc, r'z_{i,m}(f)=Y_{i,m}(f)\,Y_{i,1}^{*}(f)=|A_i(f)|^2\,H_m(f)H_1^{*}(f)\,e^{-j2\pi f(\tau_{i,m}-\tau_{i,1})}', 'eq_cross')
    P(doc, '它与接收钟差 δ、钟漂 η（一阶）、基站同步误差 ε_i、基站硬件响应 G_i(f) 全部无关。对中心对称阵列（UCA 满足 Σ_m ∂τ_{i,m}/∂θ = 0），'
           '丢弃各基站的公共相位不损失任何方位 Fisher 信息；单基站方位信息中载波项占 f_c²/(f_c²+B²/12) ≈ 98%，这正是只用包络的方法丢掉的部分。')
    EQN(doc, r'J_{\theta,i}=\mathrm{SNR}_i\,N\,M\left(\frac{2\pi R\cos e_i}{c}\right)^{2}\left(f_{c,i}^{2}+\frac{B^{2}}{12}\right)', 'eq_fisher')
    P(doc, '稀疏阵列的栅瓣在单基站上不可消除，但不同基站的方位、俯仰与子带各不相同，其栅瓣在位姿空间 (x, y, ψ) 中互不对齐。定义位姿域模糊函数')
    EQN(doc, r'\Lambda(\Delta\mathbf{p},\Delta\psi)=\frac{1}{I}\sum_{i}\mathcal{A}_i\!\left(\theta_i(\mathbf{p}_0+\Delta\mathbf{p},\psi_0+\Delta\psi);\,\theta_i(\mathbf{p}_0,\psi_0)\right)', 'eq_ambig')
    P(doc, '4 个基站时最高假峰降到 0.70–0.84，且都离真值 0.6–8 m 远，可被 cm 级 TDOA 门控排除；给定位置后对航向全搜索并做 ratio test。'
           '由此得到**口径–模糊折中**：精度 ∝ 1/(f_c R)，模糊余量随 kR/M 增大而耗尽（图 7b）。本阵列 R = 50 mm 处于有利区间。')
    P(doc, '**说明**：跨阵元相位与发射端时钟无关在干涉测向中是已知事实；本贡献的新意在于把它与位姿域模糊函数、多基站耦合、口径–模糊折中结合，'
           '形成稀疏 UCA 下可用的完整估计器，并作为 CIPHER-X 的中间层与独立完好性检验。')

    H(doc, '3.2  C2：宽松同步即载波同步——交叉点与敏感度（命题 1、2）', 2)
    P(doc, '射频（2.4 GHz）载波周期 0.4 ns，载波级同步需要 ps 级；声学（20 kHz）载波周期 50 μs，**1 μs 的同步残差只相当于 7.2° 相位、0.34 mm 距离**。'
           '于是跨基站绝对载波相位可直接使用：把直达径载波相位与接收时刻绑定，')
    EQN(doc, r'\tau_{\phi,i}+b_i+\frac{n_i}{f_i}=t_{rx}+(1+\eta)\,\tau_{c,i}(\mathbf{p};c)+\varepsilon_i,\qquad \varepsilon_i\sim\mathcal{N}(0,\sigma_s^2)', 'eq_carrier')
    P(doc, '其中 τ_φ,i 为载波时刻（模 1/f_i），b_i 为每基站相位常数（标定），n_i 为整周数，τ_c,i 为 chirp 中心历元的传播时延。')
    P(doc, '**命题 1（交叉点）**：阵列给出的位置精度 σ_A 与跨基站载波给出的 σ_X ≈ κ c (σ_s² + σ_c²)^{1/2} 相等处')
    EQN(doc, r'\sigma_s^{\star}\approx\frac{\bar{\rho}}{2\pi f_c R\cos e\;\kappa c\,\sqrt{N M\,\mathrm{SNR}\,I/2}}', 'eq_cross_pt')
    P(doc, '即容许的同步误差 ∝ 基线比 ρ̄/R 除以载波频率；本系统约 6 μs。σ_s ≪ σ* 时位置由跨基站载波决定，σ_s ≫ σ* 时自动退化为 CIPHER（图 3a）。')
    P(doc, '**命题 2（敏感度）**：每阵元相位误差 δχ 在阵列模式下引起 O((λ/2π)(ρ̄/R) δχ/√M) 的位置偏差，在载波模式下只经阵列平均相位进入，为 O(κ(λ/2π) δχ/√M)，'
           f'二者之比为 O(ρ̄/R)。数值：每阵元 5° 误差 → 阵列模式 {e12["array_only"]["bias5deg_mm"]:.1f} mm，载波模式（σ_s = 1 μs）{e12["tied"]["1us"]["bias5deg_mm"]:.2f} mm（图 3b）。'
           '推论：载波模式下位置几乎不依赖阵元标定，航向仍由阵列决定。')
    FIGR(doc, 'fig05_crossover.png', '命题 1、2 的数值核对（exp12）：(a) 位置 CRLB 随同步残差的交叉；(b) 每阵元 5° 相位误差引起的位置偏差。')

    H(doc, '3.3  C3：声速可观性与"多径即温度计"（命题 3）', 2)
    P(doc, '单帧未知量 (p, t_rx, s)（c = c₀(1+s)），观测 t_i = t_rx + d_i(p)(1−s)/c₀ + ε_i。s 可辨识当且仅当 [d_1..d_I] 不在 span{1, u_x, u_y} 内；'
           '**4 个基站时方程恰定、无冗余**。两个推论：(a) 同步约束无法单帧检出坏基站（实验中同步残差 RAIM 恒为 0）；(b) 声速必须另有信息源。'
           '时延域多径模型中，镜像径与直达径的时延差 Δd_iℓ/c **与同步无关**，单帧即可把 s 估到 ±0.017%（≈ 0.1 °C）。')
    P(doc, '这条结论也修正了一个此前被忽略的风险：若声速只进入同步项，模型声速偏 0.3%（约 0.5 °C）会使镜像径时延偏 12 μs（20 kHz 下约 90° 载波相位），'
           'CIPHER 从 3.5 mm 退化到 21 mm；把声速放进完整几何后不受影响（3.46 mm）。')

    H(doc, '3.4  C4：整周可固定性由声速精度主导（命题 4）', 2)
    P(doc, '浮点整周的协方差中，可被 (p, t_rx, s) 吸收的方向包含 s 方向 ∝ [d_i f_i]：错误的整周组合可沿此方向以 δs ≈ λ/d̄ 的声速变化"补偿"。'
           f'声速先验 ±0.3% 时单帧固定成功率仅 {e12["ambiguity"]["4 anchors, s prior 0.003, K=1"]:.2f}，20 帧 {e12["ambiguity"]["4 anchors, s prior 0.003, K=20"]:.2f}；'
           f'声速精度 0.01% 时单帧 {e12["ambiguity"]["4 anchors, s prior 0.0001, K=1"]:.2f}、5 帧 {e12["ambiguity"]["4 anchors, s prior 0.0001, K=5"]:.2f}；'
           '增加到 6 个基站几乎没有改善（图 4a）。**设计含义：先用多径模型把声速估准，再固定整周**——这与 GNSS 中"先估电离层"的思路一致，但在声学中声速可由同一信号的多径几何直接得到。')

    H(doc, '3.5  C5：FDMA 宽巷别名与同步上限（命题 5）', 2)
    P(doc, '最难区分的错误候选是"公共平移"：t_rx 移动一个平均周期 1/f̄、全部整周 +1。由于各基站子带中心频率不同，它留下失配')
    EQN(doc, r'm_i=\frac{1}{f_i}-\overline{\left(\frac{1}{f}\right)},\qquad \Delta\chi^2\approx\eta\,K\,\frac{\Vert\mathbf{m}\Vert^2}{\sigma_s^2}\ \ \Rightarrow\ \ \sigma_{s,\max}\approx\Vert\mathbf{m}\Vert\sqrt{\eta K/\gamma}', 'eq_alias')
    m = e12['alias']['m_us']
    P(doc, f'本系统 ‖m‖ = {sum(x*x for x in m)**0.5:.1f} μs，未被位置/声速吸收的比例 η ≈ 0.15–0.3；K = 10 帧、检验门限 γ = 8 时 σ_s,max ≈ 4 μs，'
           '与实验中经验设定的门限一致（图 4b）。**设计含义：基站中心频率的离散度决定可容忍的同步误差**——FDMA 频带分配不仅是多址手段，也是模糊度设计参数。')
    FIGR(doc, 'fig06_ambiguity.png', '命题 4、5 的数值核对（exp12）：(a) 整周固定成功率与声速先验、帧数、基站数；(b) 公共平移别名的可检测量随同步残差的变化。')

    H(doc, '3.6  C6：运动阵列的载波观测与棋盘式波形', 2)
    P(doc, '**距离–多普勒耦合**：接收机以径向速度 v_r 接近时，匹配滤波输出的载波相位一阶上对应 chirp 中心历元时延 τ_c，而包络对应 τ_c − (v_r T/c)(f_c/B + 1/2)，'
           '偏移被放大约 13 倍。**波前采样像差**：运动阵列在不同时刻采样同一波前，阵元间时延差为同一时刻几何值的 1/(1+β)。')
    P(doc, '**二阶修正（本项目发现）**：载波"跟踪 chirp 中心"只在一阶成立，还存在一个 β² 量级、符号随扫频方向变化的项（2 m/s 时 0.13–0.21 周）。它是模板自身的性质，可精确扣除：')
    EQN(doc, r'\tau_c=-\frac{\arg\!\left(c_{i0}\,e^{j\varphi_{pk}}\right)}{2\pi f_c}+\left[\frac{T}{2(1+\beta_p)}+\frac{\varphi_{pk}(\beta_p)}{2\pi f_c}\right]', 'eq_beta2')
    P(doc, '方括号由先验速度下的模板直接算出，对速度误差只有二阶敏感。修正后 2 m/s 时载波观测对真值的抖动从 0.04–0.05 周降到 0.004–0.009 周。')
    P(doc, '**棋盘式上/下扫波形**：每个基站同一时刻只能发一种信号，但可逐帧改变；令基站 i 第 k 帧的扫频方向为 (−1)^(i+k)（图 5）。相邻基站与相邻帧方向相反，'
           '耦合项正负抵消：两帧联立即可求出 (x, y, v_x, v_y, t_rx)，运动中冷启动从第 1 帧起可用；接收钟漂也因此在单帧内可观（并被估计出来）。')
    FIGR(doc, 'fig02_waveform.png', '棋盘式上/下扫 FDMA 波形：4 个基站分占 17–23 kHz 的 4 个 1.5 kHz 子带，扫频方向按 (−1)^(i+k) 交替。')

    # ---- 4 framework
    H(doc, '4  算法框架', 1)
    FIGR(doc, 'fig03_framework.png', '算法总框架。CIPHER 每帧独立运行，并把位姿、协方差、t_rx 与声速交给 CIPHER-X（橙色箭头）；CIPHER-X 的固定解反馈为 CIPHER 下一帧的先验（蓝色箭头）。')
    H(doc, '4.1  信号与前端', 2)
    B(doc, '**波形**：4 个子带（17–18.5 / 18.5–20 / 20–21.5 / 21.5–23 kHz），40 ms chirp、10% Tukey 窗、100 ms 帧，棋盘式扫频方向；基站播放文件由 firmware/make_anchor_waveforms.py 生成（48/96 kHz）。')
    B(doc, '**前端**：每个子带用上、下扫两种参考脉压，较强者即该基站本帧的扫频方向；首径检测带 8 ms 回溯（防止地面反射与直达径相消时把墙面反射当首径）；扣除麦克风谐振的标称群时延。')
    B(doc, '**自主帧定时**（pipeline.frame_starts）：在第一个帧周期内捕获各基站最早到达，把窗口起点放在最早到达之前 5 ms；之后每帧前推 T_f，并以一阶环路把最早到达保持在 5 ms 处。完全不需要接收钟信息；钟漂 300 ppm、钟差未知时仍正确。')
    B(doc, '**时延窗**：直达径附近 −2…+4 ms 的基带信号；基带仅 1.5 kHz 宽，可 8 倍降采样而不损失精度（exp09）。')
    H(doc, '4.2  CIPHER：每帧的时钟免疫解', 2)
    B(doc, '**S1 门控**：跨基站首径 TDOA 闭式解；棋盘波形下用相邻两帧联立消除多普勒偏差；留一法应对某基站首径漏检。')
    B(doc, '**S2 模糊消解**：在门控位置邻域对航向全搜索位姿域相干度 Λ，取全局峰并做 ratio test；跟踪时只在先验处初始化，不再重新选瓣。')
    B(doc, '**S3 时延域几何感知多径模型**：对基站 i、阵元 m，')
    EQN(doc, r'r_{i,m}(t)=\sum_{\ell=0}^{L_i-1}c_{i\ell}\,g_{i\ell m}\,p_{i,\beta_{i\ell}}\!\left(t-t_{0,i}-\Delta\tau_{i\ell m}(\mathbf{p},\psi,s)+\kappa\Delta\beta_{i\ell m}\right)e^{-j2\pi f_{c,i}\Delta\tau_{i\ell m}}', 'eq_lag')
    P(doc, '路径集合 = 直达 + 程差 < 1.6 m 的房间镜像源（地面、天花板、近墙及其组合，最多 10 条）+ AGV 车体反射（深度由标定估计）。只有 Δτ 携带位姿信息；'
           '每基站一个 t_0,i（吸收全部时钟与同步误差）和每条路径一个复增益 c_iℓ（吸收幅度、喇叭响应、反射系数）。c 用最小二乘闭式消去（变量投影），'
           '对 (x, y, ψ, t_0,i, t_rx, v, ω, η, s) 用 Levenberg–Marquardt，一步按残差重估各基站方差。拟合得到的 t_0 构成同步约束 t_0,i = t_rx + d_i/c + ε_i。', indent=True)
    B(doc, '**跟踪与失锁**：恒速/恒角速预测，先验中心取上一帧帧内的 v、ω 估计（能跟上偏航角加速度）；速度不合理、跳变 > 0.5 m 或稳健相干度（去掉最差基站）连续两帧 < 0.88 时判为失锁，下一帧冷启动并重新做两帧门控。')
    H(doc, '4.3  CIPHER-X：多帧跨基站载波', 2)
    B(doc, '**X1 载波观测**：直达径复增益乘以模板峰值相位，读出 chirp 中心到达时刻（式 7 含 β² 修正）；载波时刻噪声 0.1–0.45 μs（0.03–0.15 mm）。')
    B(doc, '**X2 状态**：声速 s 为随机游走状态，由 CIPHER 每帧的估计喂入（但不确定度下限 0.01%，更精细的值只能由已固定的载波解给出）；σ_s 由残差在线自估；窗口内接收钟漂 η 为公共参数（注意 CIPHER 的 η 是等效多普勒，符号与钟频偏相反）。')
    B(doc, '**X3 捕获**：10 帧窗口，帧间整周由时间差分载波相位传递（用上一帧最优位置 + 多普勒速度预测距离变化），t_rx 按基站固定帧周期 t_rx,k = t_rx,0 + τ_k(1+η) 绑定；'
             '在 CIPHER 位置处线性化后，所有整周候选（每基站 ±1 周 × 公共平移 ±3 周）共享一个投影矩阵，枚举代价极低；以差值检验（≥ 8）接受。')
    B(doc, '**X3 保持**：逐基站携带整周数，每帧解 (dp, t_rx, s)；残差 > 4σ 的基站（周跳、直达径被挡）剔除，恢复后在亚毫米解处取整重固定；已知整周的基站少于 3 个时放弃保持。')
    B(doc, '**完好性**（GNSS 式，长时间仿真中迭代得到）：CIPHER 始终作为独立检验，连续两帧与载波解矛盾即放弃保持（不得用载波预测代替其先验后再"确认"载波解）；'
             '窗口平均新息 > max(6 mm, 4σ/√n) 时放弃（错误整周可自洽地偏 10–20 mm）；σ_s > 4 μs 时不进入载波模式；失锁后声速状态回退到 CIPHER 级不确定度。')
    B(doc, '**载波辅助跟踪**：固定且健康时，CIPHER-X 的解作为 CIPHER 下一帧的先验。')
    H(doc, '4.4  标定流程（动捕真值，一次）', 2)
    B(doc, '**阵元**：每阵元面内声孔偏移 + 每（子带, 阵元）复增益与包络群时延；误差随帧数约 1/√N 下降，≥ 50 帧接近模型下限（exp06）。')
    B(doc, '**AGV 车体反射深度**：对候选深度（0.02–0.40 m）各拟合一次时延域模型，取残差最小者；改善 < 1 dB 判为"无车体反射"（exp11：5 cm → 0.06 m，12 cm → 0.12 m）。')
    B(doc, '**每基站载波常数 b_i**：与标定期间的声速联合估计，逐帧公共偏移用圆周网格搜索、b_i 用圆周平均（避免 ±半周回绕）。')
    B(doc, '**阵元编号手性**（PCB 顶/底视不确定）：两种编号假设下比较相干度，判别正确率 100%。')
    H(doc, '4.5  嵌入式实现与计算量', 2)
    rows = []
    for k, lab in (('decim1_paths10', '不降采样，10 条路径'), ('decim8_paths10', '8× 降采样，10 条路径'),
                   ('decim8_paths6', '8× 降采样，6 条路径'), ('decim8_paths4', '8× 降采样，4 条路径')):
        rows.append([lab, f'{e9[k]["pos_rmse_mm"]:.2f}', f'{e9[k]["psi_rmse_deg"]:.4f}'])
    TABLE(doc, '复杂度削减对精度的影响（exp09，10 个静止位姿，跟踪模式）', ['配置', '位置 RMSE (mm)', '航向 RMSE (°)'], rows, [6.0, 4.0, 4.0])
    P(doc, '结论：时延窗 8× 降采样无损；镜像路径至少保留 6–10 条。按 C/NEON 单精度、有效 0.3 GFLOPS/核估算：单次残差约 0.6 MFLOP，跟踪模式 5–10 次 LM 迭代 × 解析 Jacobian，'
           '每帧 40–80 MFLOP，T113 双核约 0.1–0.15 s/帧 → **5–10 Hz 可行**；CIPHER-X 第二级 < 5 ms。')

    # ---- 5 simulation
    H(doc, '5  仿真验证', 1)
    P(doc, '仿真器逐采样精确合成：运动（含多普勒与时间压缩）、接收钟差与钟漂、基站同步抖动、逐面反射系数的镜像源混响（2 阶）、喇叭活塞指向性、麦克风谐振与失配、PDM L/R 采样偏移、'
           '自噪声与环境噪声；可选加入估计器不建模的家具散射体、AGV 车体反射板与各路径声速差。所有实验结果在 sim/results/ 中，可由对应脚本复现。')
    H(doc, '5.1  静止精度、口径与鲁棒性（CIPHER）', 2)
    FIGR(doc, 'fig07_static_aperture.png', f'(a) 混响房间单帧静止精度（exp01，32 位姿 × 3 帧）：CIPHER {rev["CIPHER"]["pos_rmse"]:.1f} mm、0% 粗差，多径感知 CRLB {crl:.1f} mm；(b) 口径–模糊折中（exp05）。')
    FIGR(doc, 'fig08_cipher_robustness.png', '(a) 基站同步抖动的影响（exp02）；(b) 运动中起步时运动模型的作用（exp03）；(c) 阵元失配的动捕标定（exp06）。')
    P(doc, f'要点：CIPHER 对接收钟漂 ±100 ppm 免疫（{e2["drift"]["-100ppm"]["CIPHER"]["pos_rmse"]:.1f}/{e2["drift"]["+0ppm"]["CIPHER"]["pos_rmse"]:.1f}/{e2["drift"]["+100ppm"]["CIPHER"]["pos_rmse"]:.1f} mm），'
           '同步抖动 ≤ 50 μs 时平滑退化；不建模运动时 1 m/s 下误差上百毫米，建模后 3–8 mm；综合阵元失配（谐振离散 2%、相位 5°、声孔 0.3 mm）未标定 40 mm，48 帧标定后 6.6 mm（模型下限 6.3 mm）。'
           '（exp01–06 在声速已知、帧定时用真值的条件下运行；exp07 起全部使用声速估计与自主帧定时。）')
    H(doc, '5.2  CIPHER-X：运动中的毫米级精度', 2)
    FIGR(doc, 'fig09_carrier.png', '运动场景（exp07，8 条轨迹 × 20 帧，棋盘波形运动中起步）：整周固定帧上 CIPHER-X 与同一批帧 CIPHER 的对比。', 13.5)
    TABLE(doc, 'exp07 汇总（模型声速偏 0.3%，帧定时自主获取，σ_s 自估、初值 4 μs）',
          ['场景', 'CIPHER', 'CIPHER-X 全部帧', '固定率', '固定帧 X / CIPHER', '错误固定', 'σ_s 自估'],
          [[k.replace('_sync', '，同步 ').replace('us', ' μs').replace('v', ''), f'{v["cipher_pos"]:.2f} mm', f'{v["x_pos"]:.2f} mm',
            f'{100*v["fixed"]:.0f}%', (f'{v["fixed_x_pos"]:.2f} / {v["fixed_cipher_pos"]:.2f} mm' if v['fixed'] > 0 else '—（保持 CIPHER）'),
            f'{100*v["wrong_fix"]:.0f}%', f'{sorted(v["sigma_s_final_us"])[len(v["sigma_s_final_us"])//2]:.1f} μs']
           for k, v in e7.items()], [3.3, 1.8, 2.3, 1.4, 3.2, 1.6, 1.8], font_size=8.5)
    P(doc, '固定率受 2 s 短轨迹限制（前约 0.6 s 为声速收敛与捕获期）；同步 10 μs 时 σ_s 自估超过 4 μs，系统正确地停留在 CIPHER 模式，不产生错误固定——即命题 1、5 预言的交叉行为。')
    H(doc, '5.3  长时间运行、遮挡与恢复', 2)
    FIGR(doc, 'fig10_long.png', '30 s 长轨迹（exp08，0.5 m/s，4 条轨迹中位数）：上图位置误差，下图整周固定比例；黄色为遮挡时段。')
    rows = []
    for ph in ('clear 0-8 s', 'A1 blocked 8-10 s', 'A3 silent 15-15.5 s', 'A0+A2 blocked 22-23 s'):
        a, b2 = e8['v0.5'][ph], e8['v1.0'][ph]
        name = {'clear 0-8 s': '正常 0–8 s', 'A1 blocked 8-10 s': '基站 1 直达径被挡', 'A3 silent 15-15.5 s': '基站 3 静默',
                'A0+A2 blocked 22-23 s': '基站 0+2 同时被挡'}[ph]
        rows.append([name, f'{a["cipher"]:.1f} / {a["x"]:.1f} mm（{100*a["fixed"]:.0f}%）', f'{b2["cipher"]:.1f} / {b2["x"]:.1f} mm（{100*b2["fixed"]:.0f}%）'])
    rows.append(['遮挡结束后（剔除过渡帧）', '2.8 / **1.4** mm，2 s 内重新固定', '4.0 / **2.9** mm，2–3 s 内重新固定'])
    TABLE(doc, 'exp08 分段结果：CIPHER / CIPHER-X（固定率）', ['阶段', '0.5 m/s', '1.0 m/s'], rows, [4.6, 5.6, 5.6])
    P(doc, '两个基站同时被挡时只剩 2 个可用基站，超出 4 基站系统的能力（两种方法都失效），这符合预期；关键在于系统能检测并在 4 个基站信号恢复后迅速重新固定。')
    H(doc, '5.4  估计器未建模的效应与 AGV 车体反射', 2)
    FIGR(doc, 'fig11_mismatch.png', '模型失配（exp10）与 AGV 车体反射的标定建模（exp11）：橙色为 CIPHER 全部帧，蓝色为 CIPHER-X 固定帧。')
    P(doc, f'家具散射（−20 dB）下 CIPHER-X 仍为 {e10["clutter 6 x -20 dB"]["fixed_x"]:.2f} mm；强散射（−12 dB）使 CIPHER 本身退化，CIPHER-X 自动不进入载波模式。'
           f'**AGV 车体反射是最危险的失配**：阵列下 5 cm / 12 cm 的金属车体不建模时 CIPHER 达 {e11["deck 5 cm / unmodelled"]["cipher"]:.0f} / {e11["deck 12 cm / unmodelled"]["cipher"]:.0f} mm，'
           f'把标定估计出的深度加入模型后恢复到 {e11["deck 5 cm / calibrated depth"]["cipher"]:.1f} / {e11["deck 12 cm / calibrated depth"]["cipher"]:.1f} mm，CIPHER-X 无错误固定（代价：首次固定约晚 1 s）。'
           f'路径间温差 0.1 °C 时 CIPHER-X 为 {e10["temperature 0.1 degC"]["fixed_x"]:.2f} mm——温度均匀性是亚毫米精度的物理前提。')

    # ---- 6 comparison
    H(doc, '6  与现有方法的比较', 1)
    TABLE(doc, '方法比较', ['', '观测量', '时钟要求', '精度决定因素', '典型精度'],
          [['SCALA / A-LOAF / LUNA', '小口径测向 + 群时延', '异步', '口径（R = 1.6–2.5 cm）', '2–12 cm'],
           ['DRAPE / HARP / PALM', '逐阵元群时延 TDOA', '基站同步', '带宽 B', '3.6–5.8 cm'],
           ['CAT / MilliSonic', '单链路 FMCW 相位', '需参考点 / SFO 标定', '相对量，累积误差', '数 mm（相对）'],
           ['**CIPHER**（本文）', '跨阵元载波相位 + 门控', '**免疫**钟差、钟漂、同步误差', '载波频率 f_c 与口径 R', '**~4 mm / 0.02°**'],
           ['**CIPHER-X**（本文）', '+ 跨基站绝对载波相位', 'σ_s ≲ 4 μs（自估，否则退回 CIPHER）', '载波周期 + 同步残差', '**~1 mm**']],
          [3.4, 3.6, 3.4, 3.2, 2.6], font_size=8.5)

    # ---- 7 limits
    H(doc, '7  适用条件与局限', 1)
    B(doc, '**至少 3 个基站无遮挡**：4 个基站中两个同时被挡时无法定位（可检测并快速恢复）；如需容忍双遮挡应增加基站。')
    B(doc, '**同步残差**：CIPHER-X 需要 σ_s ≲ 4 μs（命题 5），算法会自估并在超限时退回 CIPHER。')
    B(doc, '**温度均匀性**：路径间温差 ≤ 0.1 °C 时保持亚毫米；0.3 °C 时约 4–5 mm（用户已确认现场无明显气流）。')
    B(doc, '**机器人本体反射**必须建模（深度由标定得到）；其他未建模的强散射（−12 dB）会使 CIPHER 退化。')
    B(doc, '**声学测量的固有限制**：目前全部为仿真结论；真实喇叭相位、20 kHz 以上麦克风响应、湍流等需实测验证。')
    B(doc, '**计算量**：Python 实现每帧约 1–3 s，实时运行需 C/NEON 重写（解析 Jacobian、时延降采样）。')

    # ---- 8 next
    H(doc, '8  后续工作', 1)
    B(doc, '**实测采集**（docs/experiment_protocol.md）：4 个基站同步循环播放 firmware/waveforms/ 中的文件；T113 录 8 通道原始数据；动捕记录基站与 AGV；'
             '每段首尾静止 3 s 并原地转一圈用于离线时间对齐；采集标定、静止、0.3/0.5/1 m/s 运动、遮挡四类数据。')
    B(doc, '**离线验证**：用与仿真相同的前端、帧定时、标定与估计器处理实测数据，与动捕真值比较。')
    B(doc, '**理论**：补全命题 1–5 的严格证明；**固件**：C/NEON 移植；**论文**：目标 IEEE TIM / TSP。')

    # ---- appendix
    H(doc, '附录 A  代码与文件结构', 1)
    TABLE(doc, '主要模块', ['路径', '内容'],
          [['sim/cipher/config.py', '场景配置：阵列、信号（棋盘波形）、房间、基站、时钟、麦克风、失配选项'],
           ['sim/cipher/channel.py', '逐采样信号合成（运动、时钟、同步、镜像源、指向性、谐振、散射体、车体反射、遮挡）'],
           ['sim/cipher/frontend.py', '脉压、扫频方向判别、首径检测、时延窗（可降采样）'],
           ['sim/cipher/pipeline.py', '自主帧定时（不依赖接收钟信息）'],
           ['sim/cipher/lagmodel.py', '时延域多径模型（变量投影 + LM）、多普勒模板、阵元标定、车体反射深度扫描'],
           ['sim/cipher/estimators.py', 'CIPHER（门控、模糊消解、精化）、跟踪（失锁检测、载波辅助）、对比基线'],
           ['sim/cipher/carrier.py', 'CIPHER-X：载波观测、载波常数标定、多帧载波跟踪器（捕获、保持、完好性）'],
           ['sim/cipher/crlb.py', 'Cramér–Rao 界'],
           ['sim/experiments/exp01–12', '各实验脚本（见附录 B），结果写入 sim/results/'],
           ['firmware/make_anchor_waveforms.py', '基站播放文件生成'],
           ['docs/', 'theory.md（理论推导）、review.md（审查与开发记录）、experiment_protocol.md、hardware.md、literature.md、firmware_plan.md、本文档及其生成脚本']],
          [5.2, 10.6], font_size=8.5)
    H(doc, '附录 B  实验索引', 1)
    TABLE(doc, '仿真实验', ['实验', '内容', '关键结果'],
          [['exp01', '静止随机位姿，CIPHER 与基线、CRLB', f'混响 {rev["CIPHER"]["pos_rmse"]:.1f} mm / {rev["CIPHER"]["psi_rmse"]:.3f}°，基线 54–237 mm'],
           ['exp02', '同步抖动、接收钟漂', '钟漂免疫；抖动 ≤ 50 μs 平滑退化'],
           ['exp03', '运动：棋盘 vs 上扫、运动模型消融', '运动中起步 3.2 / 4.1 / 7.5 mm（0.5 / 1 / 2 m/s）'],
           ['exp04', '阵元失配敏感性、标定、编号手性', '综合失配 49 mm；手性判别 100%'],
           ['exp05', '口径–模糊折中', 'R = 50 mm 处于有利区间'],
           ['exp06', '标定帧数与模型下限', '48 帧 6.6 mm（下限 6.3 mm）'],
           ['exp07', 'CIPHER-X 运动', f'固定帧 {e7["v0.5_sync2us"]["fixed_x_pos"]:.2f} / {e7["v1.0_sync2us"]["fixed_x_pos"]:.2f} / {e7["v2.0_sync2us"]["fixed_x_pos"]:.2f} mm，无错误固定'],
           ['exp08', '30 s 长轨迹 + 遮挡 / 掉线', '单遮挡 ≈1 mm；恢复后 2–3 s 重新固定'],
           ['exp09', '嵌入式复杂度削减', '8× 降采样无损；路径 ≥ 6'],
           ['exp10', '估计器未建模的效应', '车体反射最危险；温差 ≤ 0.1 °C'],
           ['exp11', 'AGV 车体反射深度标定', '27 / 88 mm → 4.2 / 3.8 mm'],
           ['exp12', '命题 1–5 线性化核对', 'σ* ≈ 6 μs；σ_s,max ≈ 4 μs']],
          [1.6, 6.0, 8.2], font_size=8.5)

    upd = OxmlElement('w:updateFields')                     # Word refreshes the TOC on open
    upd.set(qn('w:val'), 'true')
    doc.settings.element.append(upd)
    doc.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
