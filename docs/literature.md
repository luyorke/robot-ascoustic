# 文献笔记（papers/ 9 篇 + 网络检索）

## 1. 逐篇要点

| 论文 | 场景/同步 | 信号 | 核心方法 | 结果 |
|---|---|---|---|---|
| **CAT** (Mao, MobiCom'16) | 扬声器→手机，异步 | 正弦(多普勒) + 17–19.5 kHz FMCW | 分布式 FMCW：伪发射时刻 → 只测距离变化，参考点给绝对距离；静止时线性拟合拍频漂移斜率补偿 SFO；FMCW 距离 + 多普勒速度 + IMU 滑窗优化（凸松弛 + ADMM） | 2D 5–7 mm，3D 8–9 mm |
| **SCALA** (Liu, ZJU) | 4 个**彼此异步**基站 → 8 元 UCA (R=2.5 cm) | FDMA LFM 16–20 kHz，各 1 kHz，50 ms | FBTE：一条长参考 chirp 一次 dechirp 得所有基站拍频(CZT+二次插值+跨帧跟踪)；TABF：按 TOA 对齐后共享阶次 FrFT 滤波去多径 + MUSIC；AOA 闭式位姿 → 多帧平均估计**基站间**时钟偏置 → 齐次提升 (8 维) 的闭式 AOA–TDOA 位姿 (SVD) | 2.15 cm / 0.27° |
| **A-LOAF** (Liu, TIM'26) | 异步基站 → 8 元 UCA | 同上 FDMA | GCC 粗 TOA → 谱减去混响 → FrFT 域矩形滤波 → MUSIC AOA；帧间 FMCW 径向位移约束；SO(2) 角误差 + Huber + DRR 加权；PSO 求解 + 关键帧 | 11.7 cm / 0.60° |
| **LUNA** (Yin) | 无人机，异步 | FDMA 20–22 kHz（避开旋翼噪声） | 相干加权 PHAT-λ 角谱；噪声窗角谱减；GEVD 可靠性加权；环形残差 Huber + NM 多起点 + EKF | 8.9 cm / 0.56°（SNR −15…−30 dB） |
| **PALM** (Xi, JSEN'25) | 同步基站 TDMA → 单麦 | chirp + 连续正弦导频 | DSC-MC：不同斜率 chirp 联立 [R,v] 线性方程，无偏消多普勒；IFI：正弦导频 Hilbert 瞬时频率积分得接收间隔内相对位移（亚 mm）；补偿 SRI 后 TDOA-ML | 5.8 cm @1 m/s |
| **HARP** (Xi, JSEN'25) | **2 个同步基站** → 4 麦大阵 (39 cm) | FDMA(16–19/19–22k) + CDMA(上下扫/不同时长) 50 Hz | CE-CC 约束互相关检测；三点共线 (TPC) 事件给初始位置；CDI：伪距增量 (t_k−t_0−kT)·c 转 TOA；刚体 ML | 3.9 cm / 6.4°，50 Hz |
| **DRAPE** (Gao, IROS'26) | 4 个**同步**基站 → 8 元 UCA (R=20 cm) | 15–23 kHz 分 4 子带，上扫+下扫，30 ms | GCC 粗定位 + FMCW 首径（最小频率极值）；上下扫 [R,v] + 速度正则(MAP)；逐麦 TDOA 双曲定位 → Procrustes+内点选择刚体位姿 | 3.6 cm / 2.35° |
| **P-TDOA** (Tu, 清华) | 同步基站 TDMA 广播 → **异步移动目标**（钟偏+钟漂） | 通用 | 两对收发时间相比消去钟偏/钟漂参数；TDOA 建模为时间多项式；连续帧差分 (STDS) 构造满秩方程；MWLS；接近并发测量 CRLB | 仿真 |
| **FDME-RATE** (Jia, IoTJ'24) | 单链路测距 | 17 kHz 起 chirp + Blackman 窗 | GCC 粗定位 → 提前 τa=10 ms 开窗 FMCW → 阈值找首径 → 自适应 MUSIC（迭代调信源数）超分辨 | 30 m 内 0.16 m |

## 2. 共性技术积木
- **信号**：近超声 LFM，FDMA 分带（每基站 0.5–3 kHz），保护间隔抑制混响；Blackman 窗降低可闻性。
- **TOA**：GCC 粗定位 → FMCW dechirp → 首径检测（阈值/最小极值/MUSIC 超分辨）→ 抛物线插值。
- **多普勒**：上/下扫或多斜率 chirp 联立消 R–v 耦合（DRAPE、PALM）；正弦导频测速（CAT、PALM-IFI）。
- **异步**：伪发射时刻 + 参考点（CAT）；伪距增量 CDI（HARP）；比值消钟参数（P-TDOA）；AOA 反推钟偏（SCALA）；静止拟合 SFO 斜率（CAT）。
- **阵列**：MUSIC/FrFT 滤波（A-LOAF/SCALA）、相干加权 PHAT（LUNA）、刚体 Procrustes（DRAPE）。
- **位姿求解**：闭式齐次 SVD（SCALA）、PSO（A-LOAF）、NM（LUNA）、ML/LS。

## 3. 观察到的空白（本项目切入点）
1. 所有工作只用了**群时延/拍频（包络）信息**，精度受带宽 B 限制（c/B ≈ 17–34 cm 分辨率）；载波相位只在单链路相对跟踪中用过（CAT 多普勒、PALM-IFI、MilliSonic），从未在"多基站 + 阵列"系统中做**绝对**相位定位。
2. "同步基站 + 异步接收机"在结构上与 **GNSS** 完全同构（卫星同步、接收机钟差+钟漂未知、多普勒、多天线测姿），但 GNSS 的双差、整周模糊度固定（LAMBDA / C-LAMBDA）、钟漂作为状态等方法没有被系统迁移到声学。
3. 本阵列口径 100 mm ≈ 5λ，相邻间距 ≈ 2λ：对相位 DOA 是灾难（栅瓣），但对**模糊度固定后**的定姿是巨大优势（基线越长，相位定姿越准）——别人用小口径(1.6–2.5 cm)规避的问题，恰好是我们可利用的。
4. SFO 在运动中无法用"静止拟合"标定（CAT），需要与速度联合可观（GNSS 中钟漂与速度联合解）。

## 4. 网络检索结论（2026-10-15）
- 未检索到"声学 + 多基站 + 麦克风阵列 + 载波相位整周模糊度固定"的工作。
- GNSS 侧可借鉴：双差消钟、C-LAMBDA / MC-LAMBDA 单历元定姿（Teunissen, Giorgi）。
- 声学侧最接近：MilliSonic（FMCW 相位跟踪，2.6 mm 3D，单扬声器）、C-FMCW、FaceOri（线性拟合钟漂）。

参考链接：
- GNSS 载波相位定姿 / MC-LAMBDA：<https://gnss.curtin.edu.au/wp-content/uploads/sites/21/2016/04/Giorgi2012GNSS.pdf>
- C-LAMBDA：<https://www.researchgate.net/publication/225472903_A_constrained_LAMBDA_method_for_GPS_attitude_determination>
- 室内 OFDM 载波相位 EKF 模糊度：<https://www.mdpi.com/1424-8220/21/20/6731>
- FaceOri：<https://arxiv.org/pdf/2203.10553>
- C-FMCW：<https://web.comp.polyu.edu.hk/csyqzheng/papers/C-FMCW_Ubicomp18.pdf>
