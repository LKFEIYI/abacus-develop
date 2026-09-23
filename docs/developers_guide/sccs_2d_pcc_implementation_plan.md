# ABACUS 2D-PCC + SCCS 实施方案

日期：2026-09-21。状态：M5 的 LCAO Gamma、多 k 点、力及网格/轨道扫描已完成；
盒高已扫描到 40 bohr，但总能尚未达到发布阈值，M5 尚未关闭。

目标：在现有原生 SCCS 上增加面内周期、法向开放的 slab 静电修正，
使修正参与电子 SCF 和溶剂极化自洽，并以一致的能量、电子势和原子力验收。
用户已明确同时覆盖中性和带电体系，并固定 y 为真空/开放方向。
交付分中性、裸带电验证、带反电荷界面三个阶段；中性阶段通过不代表整个方案完成。

## 1. 范围与物理边界

全方案固定第二晶格方向为开放方向：a、c 在 x–z 平面内，b 平行 +y。
允许 a、c 不正交，例如六方表面；要求 b 同时垂直 a、c。
不要求立方盒，不开放任意倾斜法向。面内使用正常 k 点采样，y 向仅 Γ，
即沿第二倒格矢只有零分量，规则网格通常为 Nx×1×Nz 且 y 偏移为零。
保留 ABACUS 既有 z 向实空间网格分布，不为匹配开放方向而更改 FFT 分布。

| 阶段 | 支持对象 | 验收口径 |
| --- | --- | --- |
| 首版 | 中性对称 slab、中性非对称 slab、表面吸附 | 总能/能量差、两侧远场电场、势阶跃、力随盒高收敛 |
| 必需的带电阶段 | 无反电荷的带电平板，固定电荷量 | 验证给定势规范下的场、极化电荷及参考模型；不宣称有限的无限空间场能 |
| 电极应用阶段 | 带反电荷的电极界面 | 明确反电荷位置/分布、整体中性条件、能量泛函及参考电势后单独验收 |

纯 SCCS 是介电模型，不自动提供电解质离子屏蔽。带电无限平板在均匀介电
远场仍有非零电场，因此全空间场能发散；固定势规范下的电荷—势能不能直接
视为包含远场储能的绝对总能。不得用中性 slab 的“加大盒高后总能成为常数”
作为裸带电 slab 的验收标准。[Andreussi–Marzari 2014，III.C](https://arxiv.org/abs/1410.7948)

当前保持 CPU、模守恒赝势和 nspin=1/2 范围，支持 `calculation=scf`
及 `calculation=relax` 的固定晶胞结构优化。每个电子 SCF 迭代在屏幕输出
SCCS 内迭代步数、该次 SCCS 求解耗时和当前总溶解能（Ry）。
可用 `sccs_start_drho` 延迟冷启动首轮 SCCS；达到密度阈值或
`sccs_start_nmax` 后永久开启、清空电荷混合历史，并至少继续一次电子迭代。
应力、变胞、分子动力学、恒电势、Poisson–Boltzmann、外加电场和已有 gate
模型的组合均未开放。

## 2. 已核对的代码基线

以下为本次源码检查结果；既有 0D 验证见
[原 SCCS/0D-PCC 方案](sccs_pcc_implementation_plan.md)，不将其等同于 2D 验收。

| 现有位置 | 能力与本次设计的关系 |
| --- | --- |
| `source/source_hamilt/module_surchem/sccs_parameters.h` | Boundary 已加入 Pcc2d，并接受 `pcc_2d`/`pcc-2d` |
| `source/source_hamilt/module_surchem/sccs_poisson.h` | CoulombOperator 已返回势和梯度，可直接增加 2D 后端 |
| `source/source_hamilt/module_surchem/sccs_periodic.cpp` | solve_sccs_response 已接受 CoulombOperator，复用空腔及极化组装 |
| `source/source_hamilt/module_surchem/sccs_pcc.h` | 现有矩是三维偶极和二阶矩的迹；不能把 trace 当 Qyy |
| `source/source_hamilt/module_surchem/sccs_driver.cpp` | 已分派 2D Coulomb 后端，并完整登记网格、晶胞、边界及空腔状态签名 |
| `source/source_hamilt/module_surchem/h_corr_sccs.cpp` | 已接入 2D 平滑源响应和点离子/电子真空 PCC 能量及电子势，宿主入口 Ha→Ry |
| `source/source_hamilt/module_surchem/sol_force.cpp` | 已叠加非周期反应势的平滑源力和 2D 点离子 PCC 力，输出 Ry/bohr |
| `source/source_io/module_parameter/read_inp_model.cpp` | 已接受 `pcc_2d`，说明固定 +y 开放方向、中性限制及能量/势/力范围 |

具体复用：`sccs_cavity.*`、`sccs_poisson.*`、`sccs_pw_coulomb.*`、
`sccs_pw_nonel.*`、`sccs_pw_charge.*`、`sccs_pw_reduction.*`。
不能简单给现有 `PccParameters::cube_length` 换一个 Ly：
几何、多极矩、系数和单极常数项均不同。

当前极化历史只检查边界枚举及本地/全局网格点数，未完整检查晶胞度量、
网格形状和 slab 分支切面；新路径必须补齐失效条件。
`sccs_pw_charge.cpp::pw_grid_positions` 与 `PW_Basis` 已确认使用
nrxx=nx×ny×nplane、startz_current 和 nplane 的 z 向分片布局。
原 INPUT 中“PCC 离子力尚不可用”的描述与 0D 力源码不一致；M4 已按实际
验收范围修订并重新生成参数文档。

## 3. 数学契约

### 3.1 单位、电荷和法向矩

核心采用 Hartree 原子单位；n≥0 为电子数密度，物理溶质电荷
ρ_sol=ρ_ion−n。只在宿主入口转换 Ha→Ry、Ha/bohr→Ry/bohr。

定义 A=|a×c|，L=|b|=Ly，Ω=AL，s=y−y0，y0 取盒中心。
原子和网格采用相同的法向坐标分支；slab 必须完整处于切面之间，
不能分别 minimum-image 包裹各原子而切断 slab。

对物理电荷密度定义：

\[
q=\int\rho\,dV,\qquad d=\int s\rho\,dV,\qquad Q=\int s^2\rho\,dV.
\]

Q 是法向二阶矩 Qyy，不是无迹四极矩，也不是三维 trace。
网格矩按 pool 归约一次；复制在各进程上的点离子列表不能重复求和。
保留物理 q，不以减去平均密度来强制中和源。

### 3.2 2D 修正核与势规范

首版采用平面平均（G_parallel=0）修正；对一般三维 slab 密度，
G_parallel≠0 的镜像误差仍需通过增大两侧缓冲区检验。
它不是任意盒高下精确的二维截断 Coulomb 求解器。
[Dabo 等，平面平均近似及其范围](https://arxiv.org/abs/0709.4647)

平面平均的开放核本身只确定到一个常数。为与既有 0D PCC 和
Andreussi–Marzari 式 (88)–(91) 一致，带电路径采用文献规范
α1D=π/3，而不是把单个带电平面处的开放势设为零。

令 λ(s)=∫_Aρ(x,y,s)dxdy。原始开放平面核可写为
g_open,0(u)=−2π|u|/A。对应零均值周期核在 |u|≤L 内的表示为
g_per(u)=(2π/A)(u²/L−|u|+L/6)，周期延拓。
原始差核的常数为 −πL/(3A)。文献规范相当于给开放势增加
π/(3L)+πL/(3A)，从而得到：

\[
C_{\rm PCC}(L)=\frac{\alpha_{1D}}{L}=\frac{\pi}{3L},\qquad
\Delta\phi[\rho](s)=Cq-\frac{2\pi}{AL}(qs^2-2ds+Q),
\]

\[
\nabla\Delta\phi=-\frac{4\pi}{AL}(qs-d)\,\hat y.
\]

这里的 C 明确复现 Andreussi–Marzari 式 (88) 的单极势规范。
与 g_open,0−g_per 的转换量为
δC=π/(3L)+πL/(3A)；带电参考若使用源平面势为零的开放核，
必须先加 δCq 后比较势，并加 δCq²/2 后比较能量。
中性体系的修正自能不依赖 C，
但势中的 Q 常数仍须保留，才能保持完整线性对称核。

该正号文献规范与 ENVIRON 3.1.1 当前 `core_1da.f90` 的负号常数实现不同；
这是已登记的带电绝对势/能量规约差异，不在 M1 中改写。中性体系不受该差异影响。
后续与 ENVIRON 比较带电结果时，必须显式转换均匀势移和相应的 q²/2 能量项，
不能直接把未对齐的绝对势或总能作为回归参考。

不得在每次调用后对修正势单独减平均值，也不得在极化计算和能量计算中
使用不同的常数项。改变 C 会改变带电自能 Cq²/2；它不是不同净电荷能量比较中
都可忽略的常数。中性非对称 slab 可有两个不同的远场平台，
不能同时强制两侧电势为零。

对两个分布 a、b，修正双线性型为：

\[
B(a,b)=Cq_aq_b-\frac{2\pi}{AL}
 (q_aQ_b+q_bQ_a-2d_ad_b),
\qquad
\Delta E_{\rm vac}=\tfrac12 B(\rho,\rho)
=\tfrac12Cq^2-\frac{2\pi}{AL}(qQ-d^2).
\]

中性极限为 ΔE_vac=2πd²/Ω；位于原点的单个带电平面为
ΔE=πq²/(6L)。二者分别检查偶极项和文献带电规范。
按本方案物理电荷符号，点离子修正力为
ΔF_a=−Z_a∇Δφ(R_a)，纯 PCC 项只有 y 分量；
总 SCCS 力并不限于 y。公式符号最终由能量中心差分验收。
文献常用 z 法向，本实现必须将其统一映射到 y，不能遗漏 Qzz→Qyy、
梯度分量和力数组索引的转换。

### 3.3 SCCS 极化内循环

复用现有方程：

\[
\rho_{\rm pol}=(\epsilon^{-1}-1)\rho_{\rm sol}
+\frac{1}{4\pi}\nabla\ln\epsilon\cdot\nabla\phi,
\qquad
\phi=(K_{\rm per}+\Delta K_{\rm 2d})
 (\rho_{\rm sol}+\rho_{\rm pol}).
\]

每次迭代从当前总电荷重新归约 q、d、Q，将修正势与解析梯度一起加入。
修正后的非周期多项式不能整体做周期 FFT 求梯度。
沿用混合前 RMS/最大残差双阈值、收敛后最终场重算、失败不写回历史。

均匀介电测试要求 φ_ε=φ_0/ε，q_pol=−(1−1/ε)q_sol。
非均匀空腔中 d_pol、Q_pol 必须显式计算，不能除以体相 ε 代替。
允许的中性路径应在收敛后检验 q_pol≈0；不能强行归一化掩盖边界或离散误差。

### 3.4 能量、电子势与力

沿用现有表示分工，拟采用：

\[
E=E_{\rm host}^{\rm 3D}
+\Delta E_{\rm vac}^{\rm point+electron}
+\tfrac12\int\rho_{\rm sol}^{\rm smooth}
 (\phi_\epsilon^{\rm 2d}-\phi_0^{\rm 2d})dV
+G_{\rm nonel}.
\]

真空 PCC 的矩来自点离子＋电子密度，以匹配宿主 Ewald；
介电和真空参考都用相同的平滑源及 2D 核。
`smooth_vacuum_pcc_energy` 仅作诊断，不能再次加进总能量。
固定平滑源时的反应能和点离子真空修正应分别做导数验证。
若改变离子平滑表示，应重新审计表示补偿及其密度/位置导数，不能仅凭
本分解断言结果对平滑宽度不敏感，也不能再叠加文献补偿项而造成双计数。

中性开放边界下待离散验证的电子势是：

\[
v_{\rm add}=-\Delta\phi_{\rm vac}^{\rm point+electron}
-(\phi_\epsilon^{\rm 2d}-\phi_0^{\rm 2d})
-\frac{1}{8\pi}\frac{d\epsilon}{dn}|\nabla\phi_\epsilon^{\rm 2d}|^2
+v_{\rm nonel}.
\]

解析力包括点离子真空 PCC 力和平滑离子源的介电响应力。
当前 n_cav=n，非静电项通过电子势进入；将来加入原子核心模型空腔时，
必须增加空腔的显式原子导数。LCAO 仍需宿主 Pulay 项。

带电且非均匀介电的情况需另外推导非零远场带来的边界功/泛函导数，
不能仅因均匀介电测试通过就开放其 SCF 能量与力。
这项推导是带电扩展的硬性入口条件。

## 4. 代码与 INPUT 设计

新增文件均放在 `source/source_hamilt/module_surchem/`：

| 文件/接口（拟议） | 职责 |
| --- | --- |
| `sccs_pcc_2d.h/.cpp` | SlabGeometry、SlabMoments；几何验证、矩、势、梯度、双线性型、点离子力 |
| `sccs_pcc_2d_coulomb.h/.cpp` | Pcc2dCoulombOperator；包装周期 FFT 并加 2D 解析修正 |
| `sccs_parameters.*` | Boundary::Pcc2d 与配置校验 |
| `sccs_driver.*` | 显式边界几何参数；选择后端；y 向诊断与历史缓存签名 |
| `h_corr_sccs.cpp` | 构造 slab 几何、点离子/电子法向矩、宿主能量与势 |
| `sol_force.cpp` | 2D 点离子力分派，复核平滑源力在非周期反应势下的离散导数 |

不扩充全局控制。几何和配置由宿主显式传入；旧 0D 入口先用薄包装保持行为，
每步编译后再迁移调用点。历史签名至少包含：晶格矩阵、FFT 三维尺寸、
本地布局、边界类型、y0/切面和空腔参数；不匹配时 reset。

初始 INPUT 仅扩展 `sccs_boundary=pcc_2d`；不新增面内方向开关，
全方案约定 b/+y 为法向。以下是未来示例，当前版本不能运行：

```text
imp_sol          true
solvation_model  sccs
sccs_boundary    pcc_2d
sccs_preset      custom
sccs_epsilon     78.3
sccs_rho_min     1.0e-4
sccs_rho_max     5.0e-3
sccs_gamma       0
sccs_pressure    0
```

该配置用于先隔离静电验证，不表示这些参数已经为金属表面拟合。
分子 water-* 预设可以作为对照，不能直接视为经过校准的表面模型。

在 M6 带电验收前，pcc_2d 对净带电显式拒绝；M6 完成后通过明确的带电模式
开放，不能静默解释为中性或 gate 模型。始终拒绝倾斜 b、y 向非 Γ 采样、
不相容的隔离修正/外场、应力及未开放的弛豫。
几何检查应使用容差，不依赖晶格浮点数严格相等。
审计宿主对称操作，仅保留与法向边界及 slab 实际结构相容的操作。

同时更新 C++ INPUT 注册并重新生成 `docs/parameters.yaml` 和
`docs/advanced/input_files/input-main.md`。本次仅制定方案，尚无 INPUT 行为变化，
因此不手工修改这些生成文档。

### 4.1 z 向分布下的 y 向运算

沿用现有局部索引 ir=(ix×ny+iy)×nplane+iz_local，
iz_global=iz_local+startz_current。由于 a_y=c_y=0，y 只依赖 iy；
坐标采样沿用 `pw_grid_positions` 现有约定，不在 2D 模块另造半网格偏移。
M1 通过已知位置电荷及 Fourier 模式核对该坐标约定与 FFT 相位、原子坐标一致；
已有接口不构成其采样正确性的独立证据。

1. 每个进程遍历本地 nrxx，累积 q、d_y、Q_yy 三个标量，然后使用已有
   pool 归约包装器。每次 Coulomb 调用只增加常数规模通信及 O(nrxx) 工作。
2. 修正势和梯度直接逐点加入本地数组，仅更新 gradient.y；无需收集完整三维网格，
   无需转置 FFT，也无需跨进程 halo。
3. 平面平均是固定 iy 的 x–z 面：各进程先对本地 x、z 求和，
   再 pool 归约长度 ny 的数组，最终除以全局 nx×nz。
   平均密度乘 A 才是线电荷密度 λ；局部归一化不能用 nplane 代替 nz。
4. 切片最大值也必须跨 pool 取最大；两侧 y 边界不能误判成每个进程的 z 分片边界。
   多 k 点 pool 分别拥有同一实空间物理密度时，不能跨 pool 再累加这些诊断。
5. 回归测试覆盖 nz 不能被进程数整除、startz_current≠0、ny≠nz、
   非立方盒、多个 k 点 pool。显式测试 y 向偶极产生 y 力，
   x/z 向偶极不产生该法向修正，防止将并行方向当成开放方向。

现有坐标入口要求 nplane>0；首版遵循此约束并显式拒绝空分片配置。
若需进程数超过 nz，须先单独扩展零本地点数支持及归约测试。
所有通信使用 Parallel_Reduce/Parallel_Common 等已封装接口，不直接调用 MPI。

## 5. 边界诊断与可观测量

每个算例记录 A、Ly、y0、势规范、净电荷、两种源表示的 q/d_y/Q_yy、极化矩、
内循环次数/残差、真空 PCC 能量、反应能、表面/体积能，以及各项单位。

必须提供沿 y 的平面平均 n、ρ_sol、ρ_pol、φ 和 E_y；空腔边界检查额外使用每个
法向切片的最大 |ε−ε_bulk| 和电荷绝对值指标，避免平面平均掩盖局部穿越。
两侧选择具有多个网格面的均匀介电缓冲区，记录电子/电荷尾部和空腔到切面的距离。
允许空腔贯穿面内周期边界；只对法向切面施加局域性要求。

中性 slab 检查两侧 E_y→0、各自电势平台稳定及平台差稳定。
均匀介电带电模型检查两侧场的跃变
E_y(+∞)−E_y(−∞)=4πq_sol/(Aε_bulk)，在无外加均匀场的对称约定下
分别为 ±2πq_sol/(Aε_bulk)。不要把正确的带电远场当作求解残差。

输出层首版写清楚诊断；若允许用户配置诊断输出/边界阈值，再单独注册参数。
生产拒绝阈值由 M2/M5 的网格与盒高扫描标定，不凭经验固定一个“安全盒高”。

## 6. 分阶段交付与验收

| 里程碑 | 交付 | 进入下一步的条件 |
| --- | --- | --- |
| M0 数学契约 | 电荷符号、势规范、离散泛函及独立一维参考；明确与文献规范关系 | 双线性对称、势梯度、能量导数、原点不变性通过 |
| M1 2D 数学核 | 几何/矩/势/能量/点离子力及单元测试 | 立方和非立方 slab、正负电荷、偶极层全部通过 |
| M2 固定密度 SCCS | 2D Coulomb 后端、每步极化修正、边界诊断 | 均匀介电解析屏蔽、分层介电参考、非均匀空腔方向导数通过 |
| M3 中性 SCF | INPUT、driver、宿主能量/势、PW 表面案例 | 真空极限与非对称 slab 远场正确，periodic/0D 回归通过 |
| M4 力 | 点离子和平滑源力；PW 后 LCAO | 所有代表原子的 xyz 总力通过位移/网格/SCF 收敛的中心差分 |
| M5 中性发布验证 | CTest 集成案例、示例、Ly/面内 k 点/网格/基组扫描 | 达到下列目标并登记适用范围、成本及参考版本 |
| M6 必需的带电交付 | 正负固定电荷 slab，显式势规范及边界功，能量/势/力 | 非均匀空腔带电泛函导数通过；独立边值参考和渐近场正确 |
| M7 电极应用 | 固定反电荷模型、整体中性电容器/电极算例 | 包含反电荷的全部能量和势一致，电容器解析解及力通过 |

每步单独构建、验证、提交；不把整个 core 重构与物理模型扩展堆入同一提交。
新增测试命名 `test_sccs_pcc_2d.cpp`、`test_sccs_pcc_2d_coulomb.cpp` 等，
显式加入模块及 test 的 CMakeLists.txt。

### 6.1 测试矩阵

1. **纯数学**：正负电荷平面、高斯层、中性双层；B(a,b)=B(b,a)；
   q=0 的 2πd²/Ω；改变原点并相应变换 d/Q 后势和能量不变；
   固定密度电子势方向导数及点离子力中心差分。增加 A、L 独立变化，
   捕获把面积当盒长或误用三维 1/3 系数的问题。
2. **参考求解**：以一维积分/边值解作为平面模型参考，不能只用相同矩公式
   重新计算期望值。对横向非均匀密度另用逐 G_parallel 模式开放 Green
   函数参考（非零模式指数衰减），检验 PCC 近似随间距收敛。
3. **SCCS**：ε=1、均匀 ε、平滑分层 ε，以及电子密度依赖的实际空腔。
   区分固定 ε 导数测试与完整 dε/dn 测试；中性电子扰动保持电子数。
   测试混合不收敛、非有限值、网格布局/盒长变化后的缓存失效。
4. **真实 slab**：中性对称薄层、单侧吸附的非对称 slab、具有法向偶极的
   二维材料。CO/Pt 可用作文献趋势对照；赝势、结构、泛函和势规范不同
   时不能直接要求绝对能相同。先比较本代码同一设置的收敛与独立静电参考。
5. **并行/宿主**：单进程与多进程、多个 k 点 pool、不均匀 FFT 分片；
   明确电荷归约及点离子重复计数检查；PW/LCAO 分别记录数值误差。

### 6.2 初始数值目标（拟议，尚无通过结果）

| 对象 | 目标 |
| --- | --- |
| 无 FFT 的双线性/几何恒等式 | 单位尺度输入下归一化误差 ≤1e-12 |
| 固定密度能量—势/力导数 | 归一化误差 ≤1e-6，并显示至少三个位移/扰动步长的收敛区间 |
| 真实 SCF 总力中心差分 | 每个测试分量 ≤1e-4 Ha/bohr；正式可靠结果目标 ≤1e-5 Ha/bohr |
| 中性 slab 总能差/吸附能/溶剂贡献 | 最后两档缓冲区变化 <1 meV/表面原胞；大超胞按面积归一化报告 |
| 两侧势平台及势阶跃 | 最后两档变化 <1 meV，以电子势能单位报告 |
| 中性远场残余场 | <1e-5 Ha/(e·bohr)，同时证明均匀介电缓冲区足够 |
| MPI 一致性 | 固定密度能量 ≤1e-10 Ha；端到端 SCF 另按实际 SCF 噪声登记 |

每个盒高扫描保持 slab/吸附物结构、面内面积和电荷定义不变；
用至少三档两侧缓冲区、三档网格精度，独立扫描面内 k 点。
有限差分先用 0.005/0.01/0.02 bohr，再根据 SCF 噪声判断可信区间。
新增溶剂能以同盒、同设置的溶液/真空成对计算；γ、p 先置零验证静电，
再独立打开非静电项。记录金属展宽条件及比较的是一致的能量/自由能口径。

## 7. 带电体系的必需交付与电极模型

M6 不是可省略的后续建议。它至少提供同一几何的 q=0、+q、−q 三组
固定电荷 slab，记录面电荷密度 σ=q/A；面积扩展时区分固定 q 和固定 σ。
优先保持固定电子数，不自动解释为恒电势。拟增加
`sccs_2d_charge_model=neutral|isolated_charged|counterplane`，默认 neutral；
只在各模式完成验收后注册开放，具体名称在实现评审时冻结。

isolated_charged 必须输出所选势零点及带电参考定义。验证局部场、
极化、力和固定规范下的能量差，并对有限区域的场能核对解析的远场线性项；
不比较未经规范对齐的不同净电荷绝对能量。固定 ε 的一维模型先通过，
再推导/验证 ε[n] 的边界项和完整 SCF 导数；缺少任一项时该模式仍不可用。

需要实际电极充电时，建议先实现位置和宽度固定的显式反电荷平面，
使整体中性；随后再讨论电解质离子响应与恒电势控制。这是开发范围建议，
不表示固定反电荷可以替代任意实验电解质。

反电荷平面固定在 y=y_counter，沿 x–z 均匀，总电荷 q_counter=−q_sol。
位置、宽度和参考能定义必须显式可记录；其 INPUT 单位及越界检查与模式一同设计。
固定反电荷必须进入同一个静电源、极化循环、交叉能、自能约定和电子势。
明确它是否参与空腔，以及其位置固定时谁承受外力。测试电容器模型及整体
电荷守恒；不能仅放开现有 gate_flag 冲突检查来宣称完成耦合。
若最终目标是盐溶液/电双层，则另立 PB/mPB 自由能、离子可达性与电化学
参考电势设计；恒电势还需要电子库及相应热力学势。

## 8. 验证命令与本次记录

实现时复用 GNU toolchain 的 CMake/CTest 配置，但使用独立构建目录。
既有回归集合可按以下模式选择：

```bash
OMP_NUM_THREADS=1 ctest --test-dir <independent-build-dir> --output-on-failure -R 'surchem_sccs|surchem_h_corr_sccs'
```

MPI/runtime 验证按项目要求在非受限环境运行，固定 OMP_NUM_THREADS=1。
M3/M4 已记录 executable --version、`-h sccs_boundary` 和有效 pcc_2d 案例的
`--check-input`；M5 已把代表性 LCAO 输入和参考值迁入版本化集成案例。

M1 已完成势规范登记、固定 y 法向的几何构造与校验、法向矩、势、梯度、
双线性能量及点离子力。`Pcc2dGeometry` 接受 a、c 位于 x–z 平面内的
非正交面内晶格，要求 b 沿 +y，并返回 A、Ly 和统一的 y 原点。
M2 新增 `Pcc2dCoulombOperator`，每次调用从当前总电荷重新归约 q、dy、Qyy，
在周期 FFT 势和梯度上加入解析二维修正。三个矩通过一次数组归约完成；
`pcc_2d_plane_average` 按固定 iy 对 x–z 面求和，并按全局 nx×nz 归一化，
可用于 n、ρ、φ 和 Ey 的 y 向诊断。
M3 把 `pcc_2d` 接入 INPUT、driver、PW SCF、宿主能量和电子势。2D 状态缓存现在
检查局部/全局网格、nx/ny/nz、z 分片、网格坐标签名、边界、tpiba、体积元、
原点、2D 几何及空腔参数。中性系统同时核对平滑源与点离子表示的净电荷；带电
输入和非零 y 向 k 点在生产入口显式拒绝。极化净电荷不做事后归一化，验收阈值为
残差体积阈值、用户归一化阈值和体系电荷的 1e-4 相对阈值三者最大值；该相对阈值
来自 20 Ry 水分子实际空腔的 qpol=3.90997e-4，M5 必须用网格扫描继续标定。
M4 在 `sol_force.cpp` 保留现有平滑源反作用力，并对每个点离子叠加
`pcc_2d_point_charge_force`，只在宿主边界执行 Ha/bohr→Ry/bohr 转换。
本工作区 `.git` 缺少 HEAD，`git status --short` 返回 not a git repository，
因此无法记录可靠提交号或执行基于 Git diff 的治理检查；不据此修改 Git 元数据。

本次按 `toolchain/build_abacus_gnu.sh` 的配置，先 source
`/home/lyt/DFT/abacus_sccs/toolchain/install/setup`。M0 使用独立目录
`build_abacus_gnu_2d_pcc_m0`；M1 使用新的独立目录
`build_abacus_gnu_2d_pcc_m1`；M2 使用新的独立目录
`build_abacus_gnu_2d_pcc_m2`；M3、M4 分别使用
`build_abacus_gnu_2d_pcc_m3` 和 `build_abacus_gnu_2d_pcc_m4`；M5 的 LCAO
回归使用 `build_abacus_gnu_2d_pcc_m5_lcao`。GoogleTest 只复用旧构建目录已有的源码，
没有读取或写入旧构建的二进制产物。

实际执行并通过：

```bash
cmake --build build_abacus_gnu_2d_pcc_m0 \
  --target MODULE_HAMILT_surchem_sccs_pcc_2d -j 16
OMP_NUM_THREADS=1 ctest --test-dir build_abacus_gnu_2d_pcc_m0 \
  --output-on-failure -R '^MODULE_HAMILT_surchem_sccs_pcc_2d$'

cmake --build build_abacus_gnu_2d_pcc_m1 \
  --target MODULE_HAMILT_surchem_sccs_pcc_2d \
           MODULE_HAMILT_surchem_sccs_pw_charge -j 16
OMP_NUM_THREADS=1 ctest --test-dir build_abacus_gnu_2d_pcc_m1 \
  --output-on-failure \
  -R '^MODULE_HAMILT_surchem_(sccs_pcc_2d|sccs_pw_charge)$'

cmake --build build_abacus_gnu_2d_pcc_m2 \
  --target MODULE_HAMILT_surchem_sccs_pcc_2d_coulomb surchem -j 16
OMP_NUM_THREADS=1 ctest --test-dir build_abacus_gnu_2d_pcc_m2 \
  --output-on-failure \
  -R '^MODULE_HAMILT_surchem_(sccs_pcc_2d_coulomb|sccs_charge|sccs_pcc_coulomb|sccs_functional|sccs_pw_nonel|sccs_periodic|sccs_driver|h_corr_sccs)$'
OMP_NUM_THREADS=1 mpirun -np 2 \
  ./build_abacus_gnu_2d_pcc_m2/source/source_hamilt/module_surchem/test/\
MODULE_HAMILT_surchem_sccs_pcc_2d_coulomb --gtest_brief=1

cmake --build build_abacus_gnu_2d_pcc_m4 \
  --target MODULE_HAMILT_surchem_sol_force surchem abacus_std_para -j 16
OMP_NUM_THREADS=1 ctest --test-dir build_abacus_gnu_2d_pcc_m4 \
  --output-on-failure \
  -R '^MODULE_HAMILT_surchem_(sccs_parameters|sccs_pcc_2d|sccs_pcc_2d_coulomb|sccs_charge|sccs_pcc_coulomb|sccs_functional|sccs_pw_nonel|sccs_periodic|sccs_driver|h_corr_sccs|sol_force)$'
OMP_NUM_THREADS=1 ctest --test-dir build_abacus_gnu_2d_pcc_m4 \
  --output-on-failure -R '^MODULE_IO_read_input_serial$'
```

M1 结果为 2/2 CTest 通过；数学核可执行文件报告 16/16 通过，PW 坐标测试
报告 3/3 通过。除 M0 覆盖外，新增立方、非立方及面内斜晶格几何，
不支持法向的拒绝路径，以及 startz_current≠0、ny≠nz 的局部 z 分片坐标测试。
代码质量脚本扫描 4 个本阶段相关 C++ 文件，4/4 通过。
初次链接因测试目标缺少 Matrix3 实现而失败；补齐现有 `base device` 依赖后通过。
治理检查因缺失 Git 元数据失败，不是代码规则失败。

M2 结果为 8/8 相关 CTest 通过；新增二维 Coulomb 测试在单进程及两进程 MPI
上均报告 6/6 通过。
覆盖当前电荷矩重算、解析修正、平面平均、均匀介电屏蔽、平滑分层介电的一维
开放边界参考、非均匀空腔的中性密度方向导数，以及 nz 不能由分片均分、
startz_current≠0 的模拟 z 分片归约。分层介电场误差阈值为 2e-4，极化密度
误差阈值为 3e-5；完整空腔能量方向导数误差阈值为 1e-7。
生产 `surchem` 对象目标及新增源文件的独立 C++11 编译通过。代码质量脚本扫描
M2 的 7 个相关文件，7/7 通过，平均分 96.1。首次 M2 链接漏列
`sccs_pw_charge.cpp`，补齐确定性 CMake 依赖后通过；
首次数值运行显示低网格下分层介电场误差超过目标，提升参考测试网格后通过。
两进程 MPI 在受限 sandbox 内因 PMIx 无可用网络接口退出；按项目规则在非受限环境
使用同一命令重跑并通过，两个 rank 均报告 6/6。

M3 在独立目录完成 11/11 相关 CTest 和 1/1 INPUT 串行测试；二维 Coulomb 与
driver 的两进程 MPI 分别在每个 rank 报告 6/6 和 2/2。`v3.11.0-beta9` 的
帮助、输入检查、带电拒绝和 y 向非 Γ k 点拒绝均通过。10×20×10 bohr 水分子
PW-SCF 中，ε=1、均匀 ε=5 和实际空腔 ε=1.1 均收敛；ε=1.1 的最终能量为
-391.6229394687311 eV，`E_sol_el=-0.0034141252 Ry`。非均匀 ε=5 在 pcc_2d
和匹配 periodic 对照中都未达到极化固定点收敛，属于共有的 SCCS 迭代限制，
没有通过放宽参考或修改极化电荷掩盖。完整 h_corr MPI 测试夹具在每个 rank
复制全网格后仍执行 pool 归约，会重复计数；因此不将其失败记作生产分片结论。

M4 的 11/11 相关 CTest 与 1/1 INPUT 测试通过。新增单元测试覆盖 2D 点离子力
单位转换，以及非立方 y 开放晶胞中固定电子密度总静电能对所有 xyz 位移的中心
差分；后者误差阈值为 2e-4 Ha/bohr。60 Ry 实际 PW-SCF 水分子以 0.005 bohr
位移复核总力，并以同设置 periodic 作为对照：pcc_2d 相对 periodic 的力增量
在 x/y/z 的解析值分别为 -0.0034958709、0.0050124807、0 eV/Å，中心差分为
-0.0034623680、0.0049124497、约 1e-9 eV/Å，最大残差 1.0e-4 eV/Å。
pcc_2d 和 periodic 的 y 向总力各自都保留约 8e-3 eV/Å 的共同有限截断误差，
因此 M5 需要继续做更高截断能/网格扫描，但新增 2D 力增量已经与能量导数一致。
M4 的二维 Coulomb 与 driver 两进程 MPI 再次在每个 rank 报告 6/6 和 2/2。
六个生产源文件的独立 `-std=c++11 -fsyntax-only` 检查通过；测试目标按仓库当前
GoogleTest 要求使用 C++17。代码质量脚本扫描 13 个相关 C++ 文件，11/13 达到
60 分，平均 80.2；未通过的是历史长文件 `read_inp_model.cpp`（34）和
`esolver_fp.cpp`（57），本阶段 10 个 surchem 文件全部通过。治理检查仍因工作区
`.git` 缺少 HEAD 而无法生成 staged diff。

M5 调整为优先验收 LCAO，而不是继续扩展 PW 扫描。现有 LCAO Hamiltonian 已把
`surchem` 注册到实空间有效势，LCAO force/stress 汇总也把同一 `PW_Basis` 电荷网格
和局域赝势传给 `cal_force_sol`，因此无需复制 2D-PCC 求解器。新增
`tests/02_NAO_Gamma/scf_sccs_pcc2d` 作为版本化 LCAO 回归，覆盖非均匀 SCCS
空腔、2D-PCC 能量、电子势和总力；单进程生成参考后，两进程和四进程均有
5/5 检查通过，参考总能为 -466.429046662836 eV，`E_sol_el` 为
-0.0187293116 eV。另新增可直接运行的
`examples/27_imp_sol/03_lcao_sccs_pcc2d_water`。
LCAO 水分子对 H1 的 0.005 bohr xyz 中心差分中，pcc_2d 总力与能量导数残差为
1.305e-3、6.643e-3、约 3e-10 eV/Å；匹配 periodic 对照残差为
1.276e-3、6.873e-3、约 3e-11 eV/Å，说明主要误差是两种边界共有的 LCAO
基组/实空间网格误差。pcc_2d 相对 periodic 的 x/y 力增量导数残差分别为
2.9e-5 和 2.3e-4 eV/Å。

LCAO 密度网格扫描保持相同 7 au O、8 au H 轨道，并把 `ecutwfc` 从
60/80/100 提升到 120 Ry；H1 的 y 向总力中心差分残差依次为
6.643e-3、2.118e-3、1.906e-3 和 5.345e-4 eV/Å，验证该共同的 LCAO
网格误差随截断能收敛。固定 120 Ry 后，O 的同级 2s2p1d 轨道截断半径
7/8/10 au 对应残差为 5.345e-4、6.595e-4 和 5.595e-4 eV/Å；能量和力保留
正常基组依赖，但未出现 2D-PCC 特有的 Pulay 不一致。

新增 `tests/03_NAO_multik/scf_sccs_pcc2d`，用 2x1x2 网格保持 y 向 Gamma，
并以 `kpar=2` 覆盖 LCAO k 点 pool。单进程、四进程无 pool 分组和四进程
`kpar=2` 的总能、`E_sol_el` 及力一致；版本化 Autotest 在两进程和四进程
均有 5/5 检查通过。四进程参考总能为 -466.477978092084 eV，
`E_sol_el=-0.0194803285` eV。

LCAO 盒高扫描使用 120 Ry、H 8 au/O 10 au 轨道和固定 7.2 点/bohr 的网格间距，
Ly=20/25/30/35/40 bohr 的 ny 分别为 144/180/216/252/288。相邻档 pcc_2d
总能变化依次为 10.06、6.66、4.73、3.54 meV，对应 periodic 为
11.08、7.31、5.19、3.87 meV；2D-PCC 改善了盒高依赖，但 35→40 bohr 仍未达到
1 meV/原胞的发布目标。相同区间的 `E_sol_el` 变化已降到 0.340 meV，H1 y 力
变化为 1.884e-3 eV/Å。自动网格扫描曾混入不同网格间距的 egg-box 误差，因此
不用于最终盒高判据。该结果也符合当前 PCC 只修正 G_parallel=0、非零面内模式
仍需靠缓冲区衰减的设计范围；不能把 40 bohr 宣称为已收敛默认值。

M6 使用新的独立目录 `build_abacus_gnu_2d_pcc_m6_charged`。带电测试新增
非均匀空腔固定电荷的四级步长泛函方向导数，以及 LCAO 宿主中带电 2D-PCC
能量、势和固定密度总力导数覆盖；80 Ry 测试网格下 11/11 相关 CTest 通过。
生产入口已允许显式 `pcc_2d` 加净电荷，同时警告带电开放边界场能随法向盒高
线性增长，不能把不同 Ly 的绝对总能当作收敛序列。极化净电荷理论关系仍为
严格运行检查，没有通过放宽阈值开放真实案例。

真实 LCAO 验证采用 `validation/charged_slab_lcao` 的 +1 slab、
`nelec_delta=-1`、2x1x2 k 点和 `kpar=2`。源结构原先把 slab 放在开放轴分数坐标
0.057--0.317 并跨越边界，导致首个原子密度的极化电荷为 -0.0438，而理论值约
-0.9872，严格检查正确中止。验证结构改为右手 `(a,c,-b)` 晶胞，并沿 y 平移到
0.370--0.630 后，同一检查通过。四进程计算在 32 步达到
`DRHO=9.46666e-9`，总能 -19343.02773466601 eV，
`E_sol_el=-0.0490229988 Ry`，`E_sol_cav=-0.0457913349 Ry`，总耗时
860.19 s，随后完成解析力计算。

QE/ENVIRON 文件是从同一初始结构开始的 213 步弛豫，因此比较使用第一个离子步，
不混用最终弛豫结果。初始几何的 QE 非静电能为
0.11057371-0.15677585=-0.04620214 Ry，与 ABACUS 相差
0.0004108051 Ry（0.89%）；全原子力 L2 范数分别为 0.535528 和
0.5737573 Ry/bohr（7.14%）。两者使用不同赝势和基组，绝对总能、静电嵌入能及
逐原子力不作为严格等值参考。提取值记录在 `qe_reference.yaml` 和
`abacus_m6_result.yaml`。

M5 仍未完成中性体系小于 1 meV/原胞的盒高发布阈值。按当前验证安排，负电荷
案例暂缓。`validation/charged_slab_lcao/run_all_16core.sh` 已将其余 M6 验收整理为
顺序执行的 16 MPI-rank LCAO 流程：+1 基准、沿 y 的正负整网格平移、
Ly=40/50/60/70 bohr 扫描，以及每个案例的 y 向平面平均势/场后处理。
应力和变胞仍属于后续范围。

上述 16-rank 流程现已全部完成。六个 SCF 的最大最终 `DRHO` 为
9.65523e-9。50.034 bohr 基准相对早先 4-rank 结果的总能差为
3.52e-7 eV；沿 y 正负平移 0.05 个晶胞的最大总能变化为 5.21e-5 eV，
说明 MPI 分片和原点/平移处理在该精度下稳定。40/50/60/70 bohr 的总能线性拟合
`R^2=0.9959898`、最大残差 0.01295 eV；`E_sol_el` 拟合
`R^2=0.9977712`、最大残差 0.001997 Ry。各案例远场电场跳变相对
`8*pi*q/(epsilon*A)` 的误差为 2.5%--6.5%，与严格极化净电荷检查的离散精度一致。

但两个 Ly 拟合斜率分别为 -0.0162189 eV/bohr 和
-0.00358542 Ry/bohr。Andreussi--Marzari 对所选带电 2D 规范的文字结论是电势和
总能随法向盒长线性增加。当前结果确认了线性趋势、远场 Gauss 跳变、平移不变性
和并行一致性，却没有解释斜率符号差异。因此 M6 暂不关闭；下一步应分别输出宿主
3D 带背景能、点离子/电子真空 PCC、平滑源真空 PCC 和介电反应能，逐项核对
Eq. (88)--(90) 的符号、单位和势零点。负电荷案例仍按约定暂缓。

为执行该审计，`surchem::write_sccs_diagnostics(std::ostream&)` 以显式输出流记录
最终反应场能、平滑源真空 PCC 能、点离子真空 PCC 能，以及平滑溶质、点溶质、
极化和屏蔽后的二维 q/dy/Qyy；`ESolver_FP::after_scf` 只在 SCCS 后端启用时调用。
接口不引入新的全局状态，输出同时给出重构后的 `E_sol_el`（Ry）。新的独立目录
`build_abacus_gnu_2d_pcc_m6_energy_audit` 已使用 toolchain 环境和 16 线程完成
`abacus_std_para` 与聚焦测试目标编译；
`MODULE_HAMILT_surchem_h_corr_sccs` 为 1/1 通过。质量脚本扫描三个直接修改的
surchem 源/测试文件均达到 60 分，LF 检查通过。治理检查仍因 `.git` 缺少 HEAD
无法执行 staged diff。

`validation/charged_slab_lcao/run_energy_audit_16core.sh` 会按 Ly=40、50.034、70 bohr
顺序运行 16-rank LCAO，并由 `analyze_energy_audit.py` 分别拟合宿主能、介电反应能、
点源 PCC 能和平滑源 PCC 能的斜率。受限环境中的首次 MPI 启动因
`opal_ifinit/pmix_ifinit socket()` 无可用接口而退出；非受限运行请求未获批准，
因此当前只确认了脚本语法、INPUT 检查、审计版编译和聚焦单元测试，三点数值分解
尚待在普通终端执行。该失败不作为 ABACUS 数值失败。

三点能量审计随后在普通终端完成。`E_sol_el` 与
`2*(reaction+point-PCC)` 的最大重构误差为 3.7e-11 Ry。40--70 bohr 中，宿主能
斜率为 +0.0323163 eV/bohr，反应场贡献斜率为 -0.000411889 Ry/bohr，点离子
真空 PCC 贡献斜率为 -0.00314802 Ry/bohr；后者占 `E_sol_el` 负斜率约 88%。
因此该区间的下降来自正 PCC 能随约 `1/Ly` 衰减并暂时超过宿主正斜率，Eq. (88)
的势、能和力符号彼此一致，不应整体反号。

审计同时确认原实现遗漏了 Andreussi--Marzari 附录 Eq. (A2) 的二维离子形状能：
`pi*q_pol*(Q_smooth-Q_point)/(A*Ly)`。现已由
`pcc_2d_ionic_shape_energy` 加入 `E_sol_el` 并输出独立诊断；该项对已有三个密度
分别为 0.04763、0.03811 和 0.02706 Ry。平滑与点表示中的电子四极矩相消，且
二维情况下文献明确说明无附加离子力，因此该项不修改 SCF 势和力。
新独立构建目录为 `build_abacus_gnu_2d_pcc_m6_ionic_shape`；16 线程编译完成，
二维数学核与 h_corr 宿主测试 2/2 通过，7 个相关文件的质量评分全部达到 60。
`run_a2_asymptotic_16core.sh` 的三个新算例均已完成。修正版 50.034 bohr 生产结果与
旧结果加解析 A2 项相差 `1.86e-7 eV`，`E_sol_el` 相差 `4.47e-11 Ry`，确认 A2
已正确接入总能。70/90/110 bohr 的总能斜率仍为 `-0.0121092 eV/bohr`，但宿主能
斜率为 `+0.0263065 eV/bohr`；点离子 PCC 与 A2 之和在该区间仍以
`-8.415e-4 Ha/bohr` 衰减。water-neutral 的 `epsilon=78.3` 把估算的渐近正场能
斜率压低到约 `+3.36e-4 eV/bohr`，远小于尚存的 `1/Ly` 瞬态，因此 110 bohr
以内出现负的窗口拟合斜率是预期的，不能据此翻转 Eq. (88) 的符号。

利用日志中的平滑溶质与极化多极矩重构二维 PCC 双线性交叉能后，50/90/110 bohr
的反应能 PCC 部分分别为 `-0.0292901`、`-0.0154193`、`-0.0124543 Ha`，随盒长
衰减；扣除它后保留的是周期反应能趋势。这项分解未发现真空 PCC 在反应能中被
重复加入。当前数据已经验证 A2 接线和有限盒长的受控前渐近行为，但没有用数值
计算直接到达最终的正斜率区间；继续把 LCAO 真空扩展到数百 bohr 的成本不适合作为
本轮验收条件。

据此，M6 在本轮约定的正电荷 LCAO 范围内关闭：真实 +1 slab、16-rank 并行、
整网格平移、40--110 bohr 盒长、远场 Gauss 关系、能量分解、A2 接线和解析力均已有
覆盖。负电荷案例按用户要求继续暂缓，不计入本轮关闭条件；中性体系的 M5 发布阈值
仍是独立的未完成事项。
