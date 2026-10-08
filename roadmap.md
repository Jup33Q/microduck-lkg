# MuJoCo-LKG — MuJoCo 3D 视窗光场化路线图

> 目标：把 MuJoCo（mac 端）的 3D 场景实时渲染成 Looking Glass Go 光场，
> 3D 视窗与控件彻底分离：LKG Go 上是纯光场 3D，主屏只留控制/调试界面。
> 首发场景：microduck 彩排（infer_policy 机器人行走），随后收敛为通用
> `mj-lkg-viewer`（任意 MJCF 场景可用）。
> **主线：Swift + Metal 单进程原生 app**（libmujoco C API 物理 + mjvScene→Metal
> 光栅 + StreamQuilt 显示管线 + SwiftUI 控制面板）。
> 备选 B：Python + OpenGL 快速路线（仅在主线光栅器受阻或要快速验证时启用）。
> 技术储备：StreamQuilt（interlace/校准/管线形态全部固化）、
> ~/lkg-viewer-godot（P0/P1 踩坑清单）、rehearsal_mcp（pty 键盘注入）。

## 环境事实（已验证，勿重复调研）

- MuJoCo **没有 Metal/Vulkan 后端**，自带渲染只有 OpenGL（mac 走 CGL）。
  所以 Metal 渲染 = 自己写 mjvScene 光栅器（MuJoCo 出 geom 位姿/mesh/材质，
  我们只管画）。物理则直接用 libmujoco，不涉及渲染后端。
- **libmujoco 来源**（Swift 链接用，两条都已确认）：
  - venv wheel 自带 `microduck_rl/.venv/lib/python3.12/site-packages/mujoco/`
    下 `libmujoco.3.10.0.dylib` + `include/` 头文件 —— 立即可用；
  - Homebrew 有 mujoco 3.15.0（更新，要升级时切换）。
- Python 侧（备选 B / 数据准备）：`microduck_rl/.venv`（mujoco 3.10.0）。
  普通 python 即可离屏渲染（CGL），无需 mjpython；渲染前必须先 `mj_forward`，
  否则读出全黑帧。离屏 framebuffer 需在 XML 加
  `<visual><global offwidth offheight/>` 放大。
- 朴素 Python `Renderer` 逐视角渲染+回读 = 1.91 ms/view（含 glReadPixels）→
  66 视角 8 FPS，否决；备选 B 也必须单 FBO 多 viewport 零回读。
- 离轴相机：MuJoCo 相机 `cam_intrinsic`=[fx,fy,ppx,ppy,distortion] 可代码改
  （3.10 XML schema 拒绝 intrinsic 属性）；**但主线不依赖它**——Metal 光栅器
  直接用 QuiltMath 的离轴投影矩阵，语义与 StreamQuilt 完全一致。
- LKG Go 规格：1440×2560 竖屏；quilt 11×6=66 视角 4092² tile 372×682（a0.56）；
  备选 7×8=56（tile 288×512，quilt 2016×4096）。
- 校准：Bridge REST（**PUT**，`127.0.0.1:33334` 不要 localhost）→
  pitch' = pitch·(screenW/DPI)·cos(atan(1/slope))；tilt' = screenH/(screenW·slope)
  （flipImageX=1 取负）；subp = 1/(screenW·3)。StreamQuilt
  `Calibration.swift` 全套现成（含 fetchFromBridge + lkgGoFallback）。
- interlace 数学（已逐像素验证）：z=(uv.x+i·subp+uv.y·tilt)·pitch'−center →
  fract → invView 翻转 → view=floor(z·66) → tile 采样 → ACES → maxOut 0.7。
  参考实现 StreamQuilt `LKGFixedShaders.swift`；逐像素验证过的移植
  ~/lkg-viewer-godot `lkg/lkg_interlace.gdshader`。
- Quilt 约定：view 0 = 左下 tile（GL v-up）；quilt 纹理 v 向下 → 采样时 q.y 翻转。
- 参考 quilt 基线：`cd ~/Desktop/StreamQuilt && swift run -c release sq-demo -- --dump /tmp/ref_qs11x6a0.56.png`（md5 caaf1d42f528a58ecd3eeaede99aa554）。
- StreamQuilt 可复用资产：`QuiltRenderer`（HDR quilt 目标 + interlace +
  tonemap + 测试图 PSO）、`LKGDeviceWindowController`（LKG 屏无边框窗 60Hz）、
  `Calibration`、`QuiltMath`、`LKGApp`（SwiftUI 壳 + 预览窗）。

## 总体架构（主线：单进程 Swift，物理与渲染同宿，零拷贝）

```
libmujoco (C API, Swift 直接 import)
   │  mj_step（物理 = MuJoCo 本体，sim2real 语义无损）
   ▼
mjvScene（mj_forward 后：geom 位姿/mesh/材质/灯光）
   │  Swift 遍历 → Metal draw × N 视角
   │  每视角：MTLViewport = tileRect(i)（StreamQuilt 现成）
   │         投影 = QuiltMath 离轴矩阵（焦点平面 z=0 各视角一致）
   ▼
quilt rgba16Float 4092²（QuiltRenderer 现成）
   ▼
interlace（LKGFixedShaders 现成）→ LKGDeviceWindowController → LKG Go 60Hz
   │
主屏：SwiftUI 控制面板 + 裸 quilt 预览窗（StreamQuilt app 壳现成）
```

关键决策：物理/渲染/显示同进程同语言；quilt 纹理不离 GPU；
**唯一的新代码是 mjvScene→Metal 光栅器**（周级，无未知风险）。

## 阶段划分（主线）

### P0 — Swift 宿主 + libmujoco 接入

任务：
1. SPM package `mujoco-lkg`：binaryTarget 指向 venv 的
   `libmujoco.3.10.0.dylib` + `include/` 头（module map）；
   或 brew mujoco 3.15.0（版本升级时）。敲定链接与 rpath 方案。
2. 载入 microduck `scene.xml` → mj_step 跑 1000 步 → 打印步耗时与
   qpos 变化（headless，无渲染）。
3. mjvScene 走查：mj_forward 后枚举 geom（类型/位姿/mesh id/材质），
   打印统计，确认 microduck 场景的 mesh 数据齐全可读。

验收：`swift run mujoco-lkg-probe` 输出模型统计（ngeom/nmesh/njnt）+
步耗时数据 + geom 枚举清单；场景与 Python 侧 `mj_name2id` 结果一致。

### P1 — mjvScene→Metal 光栅器 + 离轴 quilt

任务：
1. mesh 上传：mjModel.mesh_* → MTLBuffer（位置/法线/UV/索引），
   材质简化为 constant/pbr-ish（先单色+法线光照，纹理后补）。
2. 光栅器：单 render pass 覆盖 quilt，逐 tile 切 MTLViewport +
   QuiltMath 离轴投影（view 0 左下；焦点平面 z=0 像素级验证）。
3. 场景状态同步：每帧从 mjvScene 读 geom 位姿更新 instance uniforms。
4. 吞吐实测：microduck 66/56 视角 ms/frame，定布局与是否半分辨率。

验收：microduck quilt dump（`--dump` PNG）构图正确、逐 tile 视角连续；
帧耗时有数据；与 StreamQuilt 参考 quilt 同机目检风格一致。

### P2 — Interlace 上屏 + 校准通道

任务：
1. 接入 StreamQuilt 显示侧：Calibration.fetchFromBridge（失败回退
   lkgGoFallback）→ lenticularUniforms → QuiltRenderer.encodeLenticular
   → LKGDeviceWindowController.show()。
2. 校准测试图模式（encodeTestPattern 现成）：整屏纯色、转头色相连续。
3. 主屏预览窗：tonemapped 裸 quilt + tile 网格。

验收（实机目检为准）：测试图无彩虹纹；microduck quilt 上屏视差正确。
离线：interlace 输出与 streamquilt `--dump-lentic` 同输入比对 ±1
（maxOut=0.7 钳制必须一致；借 lkg-viewer-godot 的 Python oracle 脚本）。

### P3 — 实时 sim 链路 + 视窗/控件分离

任务：
1. sim loop 与渲染解耦：mj_step（50Hz 策略步）与 quilt 渲染各自节拍，
   渲染跟不上时隔帧轮换视角子集（viewport 分级更新）。
2. SwiftUI 控制面板（主屏）：速度指令滑杆/按键、遥测显示；
   键盘事件走主屏窗口，LKG 窗永不抢焦点。
3. 策略推理接入：ONNX → CoreML（ANE 推理侧）或 onnxruntime CoreML EP；
   指令语义与 infer_policy.py 的 pty 键盘方案对齐（rehearsal_mcp 兼容）。
4. 性能打磨：掉帧策略（先降 interlace 输入分辨率，再降视角数）。

验收：microduck 行走彩排实机 60FPS（或明确记录的稳定降级档位）；
主屏控制 LKG 光场实时跟随；转头视差连续无跳变。

### P4 — 通用化打包

任务：
1. `mj-lkg-viewer <scene.xml> [--layout 11x6|7x8] [--half]` 通用查看器。
2. 一键上屏/下屏、Bridge 断线回退（回退校准 + 警告）。
3. README + 与 rehearsal_mcp / microduck_rl 的联动说明。

验收：任意 MJCF（demo_humanoid.xml、wheel_legged.xml）5 分钟上屏。

### P5 — 桌面封装（.app 放桌面，免终端启动）

任务：
1. 打包脚本：`swift build -c release` → 组装 `MicroDuck-LKG.app` bundle
   （executable + embedded dylibs：libmujoco / libonnxruntime，install_name
   改 @executable_path 相对引用；Info.plist + 图标 + 签名/公证按本机自用
   从简）。
2. 路径解硬编码：默认场景 XML、models/ 策略目录当前是写死的绝对路径，
   改为 app 相对（bundle 内资源或 ~/.mujoco-lkg 配置目录）+ CLI 覆盖。
3. 启动体验：双击启动 = 上次会话配置（策略/布局/鸭子数），LKG 不在线时
   仅主屏预览窗 + 警告条（不阻塞）。
4. 产物落 `~/Desktop/`。

验收：双击桌面 app 图标 → LKG 上屏 + 面板可用 + 策略驾驶（BAM 默认）
全流程不碰终端；拔了 LKG 也能纯预览启动。

## 备选架构 B — Python + OpenGL 快速路线

仅在主线 P1 光栅器受阻、或需要先于主线验证某些行为时启用：

```
infer_policy sim loop → mjr_render × N viewport（单 FBO 零回读）
   → interlace GLSL 全屏 pass → glfw 无边框全屏窗（LKG Go）
主屏：终端键盘控制（pty，rehearsal_mcp 现成）
```

- 优势：几乎无新渲染代码（mujoco Python 自带 mjr 渲染），验证成本低。
- 劣势：依赖 Apple 已弃用的 OpenGL；UI 分离靠终端凑合；
  离轴要靠 cam_intrinsic 主点偏移（3.10 只能代码改，XML 不接受）。
- 若启用：复用主线 P2 的校准/验收资产，别另起炉灶。

## 技术栈分工（2026-10-06 定案）

- **MuJoCo（C 库）= 物理**。唯一真实性基准：BAM 执行器、摩擦 DR、软接触、
  Newton 求解器都是策略训练时的语义，rehearsal 的价值就是同款动力学预演。
  不换、不重写。单环境每步 ~0.1ms，不是瓶颈。
- **Metal = 渲染/LKG 显示**（StreamQuilt 管线）。
- **MLX = 训练侧**（mac 本地训练/微调实验）。
- **ANE/CoreML = 推理侧**（策略部署推理；ONNX Runtime 路径已存在，CoreML 可作为
  低功耗备选）。
- **CUDA/HF Jobs = 正式训练**（mjlab/warp 现状，不变）。
- 已排除：MJX-on-jax-metal（缺算子，实测不可用）、MuJoCo Warp/Newton（CUDA only）、
  MLX/ANE 做物理（迭代求解器不匹配 ANE 静态图；MLX 自研=月级研究项目且破坏
  sim2real 语义）、经 Godot 显示（跨进程+跨 API 无零拷贝，且其多屏窗口在此
  面板有误报/钳制坑）。

## 验收基线

- 离线：interlace 输出 vs streamquilt `--dump-lentic`，同输入 quilt ±1
  （maxOut=0.7 钳制必须一致，否则大面积假失配）。
- 性能线：实机 60FPS；底线 56 布局全量更新不劣于 30FPS。
- 实机目检不可替代（截图回看交织图容易误判，以设备上转头视差为准）。

## 执行纪律

- 每阶段先跑验收再进下一阶段。
- 所有校准相关常量集中一处（主线复用 StreamQuilt Calibration，禁止另抄）。
- 改 shader/相机数学后用固定 quilt 输入做离线回归。
- microduck_rl 侧改动遵守其 AGENTS.md（sim2real 不变量是红线）。

## 进度日志

### BAM 执行器移植到 viewer（2026-10-07，bit 级对照验证通过）

- **产物**：`BamController.swift`（`BamM6` 常量 = bam/params/xl330/m6.json +
  infer_policy BAM_* 默认：kp_fw=200 / vin=7.4 / vin_drop_gain=0.1 / vin_min=6.0 /
  max_pwm=1.0 / 无 current limiter）+ `BamModelEdit`（load_mujoco_with_bam 的
  编译后 mjModel 等价物：gaintype FIXED gainprm=1 / BIAS_NONE / gear=1 /
  forcerange=±vin·kt/R=±0.9633Nm / ctrllimited off；受控 dof 的 damping/frictionloss
  清零、stiff solref=(-5e4,-2e2) solimp=(0.99,0.9999,0.001,0.5,2.0)、
  armature=1.8077e-3；mj_setConst；全量快照可 restore 回 XML PD）。
- **每子步 update**（200Hz，mj_step 前）：vin_eff=max(7.4−0.1·Σ|prevτ|,6.0) →
  duty=clamp((q_target−q)·kp·error_gain,±1)（error_gain=(4096/2π)/(256·885)）→
  V=vin_eff·duty → τ=kt·V/R−kt²·dq/R 写 ctrl、记 prevTorque；摩擦重写（m6
  directional+stribeck+quadratic，torque_external=−qfrc_bias+qfrc_constraint−
  摩擦约束行力）。**注意：mujoco 3.10 的 FRICTION_DOF 行 efc_id 存的是 dof id，
  而 bam 参考实现按关节 id 匹配（错位归属是参考语义的一部分，原样保留勿"修正"）**。
- **集成**：策略驾驶默认 BAM（= infer_policy 默认路径），面板「BAM 执行器」开关 /
  CLI `--no-bam` 回退 XML PD（切换重走 setupDucks）；物理子步拆为
  `PolicyRunner.stepPhysics()`，多鸭子逐只 update→step 交错（摩擦写在共享 mjModel
  上，last-writer-wins）；新增 infer_policy 同款 `[vel 1s avg] achieved/cmd` 遥测
  （body 系 fwd/lat/yaw + trunk_z，带 bam/xmlpd 标记）。
- **验证**（`scripts/bam_trace_ref.py`（Python 参考，复用 infer_policy 的
  load_mujoco_with_bam/PolicyInference）+ viewer `--bam-trace out.csv
  [--trace-ticks N] [--trace-cmd x,y,z]` + `scripts/bam_trace_diff.py`；
  alpha_walking.onnx cmd 0.3，200 拍/800 子步，同 STAND keyframe 初始态）：
  - **XML PD（--no-bam）：800 子步全程 bit-exact，max|Δ|=0**（回归无破）。
  - **BAM：前 1.25s（250 子步）机器精度一致（≤1.4e-16）**，2.5s ≤1.7e-11，
    4s ≤9e-6——np.exp/pow 与 libm 的 ULP 差经接触混沌放大，无语义差。
  - achieved/cmd：viewer BAM **+0.11/+0.30 m/s** = infer_policy BAM 参考值 ✓；
    ref/swift 两 CSV 末秒状态均值逐位相等。
- **排掉的雷**：① trace 分支漏设 `opt.timestep=0.005`（scene.xml 默认 0.002，
  首窗即分叉 4.4 rad/s 的量级）；② 参考脚本直接 new PolicyInference 时
  `use_projected_gravity` 默认 False（裸加速度计），≠ infer_policy 默认 True
  = viewer——对齐后 obs[3:6] 分叉消失。教训：**对照实验的参考侧也必须逐参数对齐
  被测语义**。

- **面板 reward 曲线周期低点排查（2026-10-08）**：用户报「奖励累计过一段时间有一个低点」。
  tfevents 全量分析结论：训练本身健康（连续 run 内稳定爬升），低点是**自动续跑块边界的
  统计伪影**——rsl_rl 的 Mean reward = rewbuffer（近期完结 episode 回报均值），每块是
  新进程，buffer 清空重积，块头几个 iter 只有「最早完结=早夭」的 episode 入账
  （正常 episode ~870 步 ≈ 36 iter 才完结，20-iter 块内完结的只有摔的）→ 每块第 2 拍
  固定下凹（实测 1.51→0.68→1.48）。**修复：块长默认 20→100**（低点频率 ÷5，
  反馈仍分钟级）。另注：iter 800/2000/12000 处回 0 是**从头新开的 run**（每次面板
  「开始训练」新开 run dir，iter 从 0 计）——resume 只续权重不续 episode 统计，
  跨 run 对比曲线要看 iteration 区间。curriculum 阶段台阶（head_pose_range 951/1426、
  head_pose_bias_weight 0→3 等）会让 reward 量级整体换挡，属预期。

### P5 ✅（2026-10-08 桌面封装验收）+ 策略接管不复位

- **`MicroDuck-LKG.app` 落 ~/Desktop**（`scripts/pack_app.sh`，69MB）：release 二进制 +
  libmujoco/libonnxruntime 放 MacOS/（二进制自带 @executable_path rpath，@rpath id 直接命中；
  onnxruntime 的 Vendor 绝对 rpath 留作开发机 fallback）+ Resources/{robot（自包含 xml+23MB
  assets）、scenes/arena_ball.xml、policies（pollen 10×ONNX）} + ad-hoc 签名 + AppIcon
  （flux-klein 生成的「光谱行走鸭」，源图 `mujoco-lkg/assets/AppIcon-1024.png`，
  pack 时 sips→iconutil 全尺寸）。
- **路径解硬编码**：`AppConfig.swift`（AppRes：bundle Resources → 开发路径回退）；
  场景 XML 支持 `{{ROBOT_DIR}}` 模板（arena_ball.xml 已 token 化，加载时替换后走临时文件
  喂 mj_loadXML）。训练区（microduck_rl repo 不在则隐藏）与 VLM 开关
  （LayaSidecar.available = venv+脚本在位）按存在性探测降级。
- **双击启动**：无 CLI 参数 → 恢复 `~/.mujoco-lkg/last-session.json`（场景/布局/策略/
  BAM/鸭子数/VLM/相机/跟随/头部）；首次启动 = 竞技场 + alpha_walking + BAM + VLM 自动开
  （demo-first，sidecar 在位才开）。有任意 CLI 参数则 CLI 全量主导（开发流不变）。
  LKG 不在线 = StreamQuilt 原生降级（preview only，不阻塞）。退出时 onWillTerminate 存会话。
- **策略接管不复位（用户要求）**：① 模式切换/进策略驾驶 = **原地站立复位**（保留当前
  xy+yaw，倒地 = quackd stand_up 原地扶起语义；多鸭子以鸭 0 为锚补位）；球只在首次/
  显式重置/贴墙静止时重布。② **换策略热切换**（真机 runtime 热换语义：61D obs 契约共享，
  物理状态不动；只重建 PolicyRunner + bam.reset() 把 qTarget 锚到当前关节位防猛拽）。
  policyPath 变更拆出独立 policyVersion（driveModeVersion 只管模式/BAM）。
- **打包坑**：iCloud 同步域拷来的资源带 xattr（FileProvider 元数据且**异步后补**），
  codesign 报 "resource fork... detritus not allowed" 且 xattr -cr 后仍会复发——
  解法：`cp -X`（不带 xattr 拷贝）从源头杜绝 + 清签重试兜底 + 全量替换 Desktop 拷贝
  （merge 会让 Finder 图标缓存滞留）。

### VLM 高层驾驶闭环（2026-10-08，laya-vision-microduck-kick 端到端跑通）

- **运行时**：`~/kimi-work/laya-work` + venv `~/kimi-work/venvs/laya`（torch 2.14 +
  transformers 5.19 + laya-vision repo clone + quackd[mujoco]==0.14.0）。MPS 推理
  **~50ms/决策**（README 参考值 4 核 CPU 0.6s），远超 1-2Hz 需求。
  - 端口勘察结论：aac6fef/laya-mlx 是**文本**决策模型（ModernBERT，不是 VLM），
    不适用；itcast666/laya-vision-onnx 是 vision.onnx+text_head.onnx 裸导出
    （无 choice 打分逻辑，基座版非 microduck 微调）——**选了官方 torch 路径**
    （laya.load_vlm 直读本地模型目录，语义=训练方）。
- **域差验证（先验坑排查，通过）**：`gen_quackd_frames.py` 生成 12 个标准 state
  （球在前/左/右/近/踢球距离/背后/不在场/偏心/转身）的 quackd 参考帧 + poses.json；
  viewer 新增 `--headcam-dump DIR --poses poses.json`（head-cam = head_camera 的
  cam_xpos/cam_xmat，**forward=col2**（上游相机 quat 非 MuJoCo 视见约定）、fovy 90、
  藏 group1/2、黑底、quackd headlight gamma 朴素光照）渲同名 viewer_*.png。
  **12/12 state 中 11 个 argmax 一致**（唯一分歧 ball_close@0.20m 是 fov 边缘退化帧，
  quackd 自己也只有 0.464）；KICK 在 0.27-0.28m 帧上正确触发（K≈0.55 ≈ quackd 0.59）。
  **结论：VLM 认得出 Metal 渲染，域差不构成阻塞。**
- **排掉的域差雷**：① MuJoCo headlight 是**沿相机前向的平行光**，不是逐像素指向相机
  （球边缘明暗差曾导致 mask 偏小/分布偏移）；② 灰格地板要 mark="edge" 0.85 网格线
  （训练帧有）；③ MuJoCo OpenGL 渲染是 gamma 空间朴素光照，不做线性化、无 tonemap
  （与 quilt 的线性+ACES 管线相反，head-cam 走独立 rgba8 pass）。
- **竞技场场景** `mujoco-lkg/scenes/arena_ball.xml`：quackd 0.14.0 复刻（2m 围墙
  rgba .55/.55/.58 + 橙球 r=0.05/m=0.05/condim 6/rolling 0.002 + 双地板 group1 蓝格
  /group3 灰格 floor_cam + headlight .3/.6/0 + 顶光；**无 keyframe**（球 freejoint
  改 nq），无 meshdir 坑——**MuJoCo include 合并时 compiler 必须放在 include 之后**，
  否则跨目录 include 丢 meshdir）。光栅器新增 sphere/box 图元支持。
- **闭环接入**：`LayaSidecar.swift`（子进程 laya_sidecar.py，stdin "FRAME"+192KB RGB /
  stdout "ACT choice pF pL pR pK ms"，单 in-flight 天然节流）。面板「策略驾驶」区
  「VLM 驾驶」开关（竞技场场景才出现）+ VLM 所见 256² 预览 + 决策概率行。
  CLI `--vlm` = 策略驾驶 + alpha_walking + VLM 开。动作映射 = quackd eval 语义：
  FORWARD vx0.3×1.0s / LEFT·RIGHT 转向 / KICK = **quackd 同款脚本推球**（0.30m/±35° 锥，
  1.2 m/s 沿朝向 + N(0,6°)−0.05 skew；quackd 自己也不用上游 kick 策略）。
- **BAM 转向坑（两连）**：quackd gait floor wz≥1.0 是 XML PD 下测的；**BAM 下 cmd 1.0
  原地踏步不转（achieved≈0）**，infer_policy+BAM 实测 cmd 1.5 → achieved 0.75。
  转向定为 ±1.5 rad/s；时长 0.3s≈13°/拍（quackd 训练数据 ~11°/拍；0.6s@1.5≈26°
  会过冲导致 VLM 左右震荡抽动——用户实机观察到的「1.5-0--1.5-0」）。
- **取景坑**：竞技场 bbox 自动取景把墙/球算进主体 → dist 11.6m 鸭子成小点
  （用户报「quilt 端奇怪」）。arena 模式取景只对 mesh geom（机器人），dist≈0.6。
- **E2E 实机**：`--vlm` 起跑 60FPS 不降；鸭子转向找球 → 走近 → KICK 命中
  （dist 0.11-0.24m 连续命中，球被踢走后重新追球）。BAM 执行器下转向
  achieved 0.7-0.8 rad/s @cmd 1.5。
- **转向受控（2026-10-08，用户实机反馈后）**：固定时长转向会过冲 → VLM 左右震荡。
  改为 VLM 只出方向（LEFT/RIGHT），转动时长 = 0.85×|bearing|/0.7（真值方位角 ÷
  achieved 转速，欠转 15% 防过冲，clamp [0.08,0.5]s）。quackd eval 同款考量：「一拍不能
  把球从死区一边带到另一边」（其 TURN_DEADBAND 20°）。**球不在视野（|bearing|>60°）
  时转向改走真值短边方向**——teacher 的「不可见就 LEFT」是标注工件（空帧分不出
  哪边更近），照搬到实机会原地无限转圈（用户实机反馈）。
- **致命 crash 排查（2026-10-08）**：决策日志 `String(format: "%-7s", swiftString)`
  ——%s 不接受 Swift String，strlen 野指针 SIGSEGV，viewer 主线程死在
  onDecision → 管道关闭 → sidecar SIGPIPE 静默死。表象：「READY 后无决策、鸭子不动」。
  教训：Swift String(format:) 里一律 %@（或先 padding()）；排查期 sidecar stderr
  落 /tmp/laya_sidecar.err（/dev/null 会把 traceback 吞了）。
- **球贴墙重布**：球 |x|或|y|>0.88 且速度 <0.02 持续 0.8s → placeBall() 重布
  （quackd 分布；用户反馈：不刷新鸭子会一直怼墙追球）。
- **头部控制（颈俯仰/偏航）**：13D 命令块 head_pose(4)=[neck_pitch, head_pitch,
  head_yaw, head_roll] home 增量，正 head_pitch=低头、正 head_yaw=左（quackd
  `look()` 语义，限位 ±35°/±60°）。**quackd 侧实测：yaw 精准（目标 53°→55°），
  pitch 飘（-16.7°→-24° 过冲；-10°→-4°；-20°→+2.5° 不稳定）——「参数成熟度」
  结论：yaw 可用，pitch 只能粗调**。面板「头俯仰/头偏航」滑杆已接。
  **注意：quackd eval 全程不动头**（训练分布=头部默认位），VLM 模式默认保持 0。
- **自动注视反射 = 否决（实测）**：头 yaw 对准球会把球摆到画面正中，抹掉 VLM
  的左右转向信号（ball_left 帧变 FORWARD 0.63）；踢球距离低头 38° 后 KICK 只剩
  0.09（训练分布里踢球帧 = 球在画面下边缘，低头居中 = OOD）。**VLM 驾驶下
  头保持默认位是正确选择**；要看得更低用滑杆手动压小角度（±10° 内决策仍正确，
  实测 KICK 0.68-0.80）。
- **球手感调校（2026-10-08 用户反馈）**：质量 0.05→0.15（身体冲撞不推飞）+
  rolling 0.002→0.004（踢出滚动 ~0.7m→~0.3m）。注意 KICK 脚本推球是固定
  1.2 m/s 不吃质量，质量只影响接触推挤。
- **墙对 quilt 隐身**：围墙移 group 4（默认 opt 不画 → quilt/预览无墙不挡视线；
  head-cam vopt 显式开 group 4 → VLM 帧里墙还在，训练分布不破）。改动后
  headcam-dump 12 state 重跑 argmax 仍全对（KICK 0.51-0.56 / RIGHT 0.59 等）。
- 工具链留档：`laya-work/{gen_quackd_frames.py, run_laya.py, laya_sidecar.py}`，
  参考分布 `frames/ref_probs.txt`。

- **头部独立控制（2026-10-08，用户建议「独立决策模型」落地）**：
  - **VLM 输入与物理脖子解耦**：VLM 帧改走**躯干锁定虚拟相机**（eye=head_camera 实位，
    朝向=R_trunk(t)·(R_trunkᵀR_cam)|home——即「脖子焊死在 home」但保留步态摇摆，
    正是 quackd 训练帧语义）。物理脖子随便动都不影响 VLM 分布（实测决策序列不变：
    扫描 49°→FORWARD 0.93→0.21m→KICK）。
  - **注视反射控制器**（VLM 驾驶时接管头部命令槽）：球在 1.3m/±69° 内 → yaw 跟踪
    （±55°，该通道精准）+ pitch 粗跟（0-25°，该通道策略跟踪本来就肉，纯装饰）；
    球不可见 → trunk 系 ±40° @0.25Hz 正弦扫描；τ=0.25s 平滑写 control.headYawDeg/
    headPitchDeg（面板滑杆实时可见）。手动模式仍归滑杆。
  - 没选 clef/第二个 VLM 做头部决策：gaze 目标在 ground truth 里是确定性的，
    反射控制器 50Hz 免费且零风险；模块独立可日后换模型。

### 策略对照 + 多 worker 彩排（2026-10-07 晚）

- **BAM 真值对照（rehearsal_mcp + infer_policy 遥测）**：`alpha_walking.onnx` BAM
  前进 achieved +0.11 m/s（cmd 0.30）真走；`m1_800` **BAM 下也 0.00 m/s**——
  踏步不位移不是 XML PD 的锅，是 m1_800 本身（里程碑策略）不走。
  LKG viewer（XML PD）行为与参考一致 = 移植无误。
- **滑轮腿适配（viewer）**：`scene_rollers.xml` + `roller.onnx` 直接可跑
  （被动轮关节由 actuator_trnid 反查兼容；resetPose 改走 STAND keyframe 以兼容
  rollers 身高差异），W 键滑行，速度很快，相机平滑跟随会滞后一拍属正常。
  行走策略走地板位移正常（alpha_walking）；两档策略都在面板 picker 里。
- **多鸭子同屏（viewer --ducks N，已建待用）**：N×(mjData+PolicyRunner+mjvScene)
  广播 WASD/方向键、质心跟随。注意顶点吞吐 ×N：4 只 56v 实测 ~47 FPS
  （>2 只建议半分辨率或降只数）。用户定调多 worker 走纯 mujoco 线，此档驻车备用。
- **多 worker 纯 mujoco 彩排（当前路线）**：N 个 mjpython 进程各自开窗
  （1:1 行走+轮滑）：
  `uv run python .venv/bin/mjpython scripts/infer_policy.py --walking <onnx>
  [--roller] --new-cmd-obs`；方向键逐窗控制（先点窗口聚焦）。
  **坑：迁出的新 venv 里 mjpython 缺 libpython 相对链接** →
  `ln -sf <uv python>/lib/libpython3.12.dylib ~/kimi-work/venvs/microduck_rl/lib/`
  已修（UV_PROJECT_ENVIRONMENT 迁移的附带损伤）。
- viewer WASD 收尾：松开按键归零（onKeyUp）；空格=随机推倒（每只各随一个方向）；
  X=清零；方向键与 WASD 双映射；S 撞 LKGApp 存图键 → StreamQuilt 新增
  `onKeyPreemptsBuiltins`（默认 false，行为不变）。

### P3 起步：策略本地推理接入 viewer（2026-10-07，XML PD 档已验证行走）

- **onnxruntime 接入**：venv wheel 的 `libonnxruntime.1.24.4.dylib` + jsdelivr 拉
  v1.24.4 官方 C 头（轮子不含头文件）。**坑：两个 -library xcframework 的
  Headers/module.modulemap 在 SPM include/ 撞名** → ORT 改走 C shim target
  （Sources/COnnxRuntime）+ `-L/-rpath` 指向 Vendor 副本。另一坑：
  CreateTensorWithDataAsOrtValue 的 shape count 是**维数（2）不是元素数**。
- **PolicyRunner.swift**：61D obs 逐字段对齐 infer_policy.py（imu_ang_vel 传感器 /
  trunk_base xquat 投影重力 / actuator_trnid 反查关节索引 / DEFAULT_POSE 偏置 /
  last_action 原始值 / 13D 命令块），50Hz 策略拍（wall-dt 累积，落后不追尾）+
  4×mj_step@0.005s（200Hz 物理），动作 = DEFAULT_POSE + raw×1.0 无滤波无裁剪，
  ctrl 直写（**XML PD = infer_policy --no-bam 语义；BAM 已于同日移植完毕并
  bit 级验证，见「BAM 执行器移植」节**）。初始态手写 qpos（z=0.125 + DEFAULT_POSE，非 keyframe）。
  首拍自检打印 obs[3..5]=(0,0,-1)（站立投影重力，逐位正确）。
- **面板「策略驾驶」区**：模式切换（自由物理/策略驾驶）、策略文件 picker
  （m1_800/m1_5_2000/smoke + models/pollen-policies 全量）、前进/侧移/转向滑杆、
  指令清零；行走时相机跟随 base xy（复用 stream 模式的跟随）。CLI `--policy [path]`。
- **验证（m1_800.onnx）**：站立稳定（|qpos| 1.4678 锁定，obs 投影重力逐位正确）；
  vx=0.3 前进实机可见步态行走（|qpos| 振荡 1.388，sim t 与墙钟 1:1，60 FPS 不降）。
- **JEV-27B-VL 评估（用户点名）**：决策模型（Qwen3.8-27B 冻结 + 108.9M 决策头，
  noul/choice/score 原语，choice 语义对口）但 ① 本机网络拿不到仓库（hf-mirror
  无此 org、直连超时）② 决策头是其私有服务路径（vLLM），Mac 上无现成运行时。
  结论：VLM 层先用 laya-vision-microduck-kick（201M，现成、任务对口、有 MLX/ONNX
  周边端口），JEV 待仓库可达且 Mac 推理路径明确再评。

### 56 views + 训练自动续跑（2026-10-07）

- **默认布局改 7×8=56 views**（2016×4096，tile 288×512；`--layout 11x6` 可切回）。
  56v 全分辨率 60.0 FPS（scene ~7.8ms + post ~2.4ms），顶点瓶颈下与 66v 差距有限，
  但三角吞吐 44.6M tris/帧（66v 52.6M）。interlace/校准走 spec 参数，零改动。
- **面板「自动续跑」**（默认开）：每块 20 iterations → 正常退出 2s 后自动
  `--agent.resume True`（mjlab 默认取最新 run 最新 checkpoint）续跑，无总上限，
  手动停或块非零退出才停。E2E：3 块连续（iteration 编号跨块延续 38/39→…），
  reward 0.40(×40) 持续爬升，LKG 直播全程 ~15 pkt/s 无中断（进程轮换时
  viewer 保留最后姿态，新进程起来即续）。
- **开源权重（2026-10-07 调查，hf-mirror 可达）**：官方
  `pollen-robotics/microduck-policies` 10×ONNX；续训检查点
  `tfrere/microduck-move-base-walk`（base/model.pt，2999 iters @4096，含
  agent/env yaml）；本地 microduck_m1_800/m1_5_2000.onnx。
  **laya 系**：laya 是 LLM/VLM 家族（thaitea/laya-vision 201M–256M 系列，
  SmolVLM 类），唯一 microduck 相关 = `thaitea/laya-vision-microduck-kick`
  （201M VLM 微调：相机帧 → FORWARD/LEFT/RIGHT/KICK 四选一高层决策，
  quackd 模拟器限定，低层仍靠 RL 策略；研究性质）。MLX/CoreML/ONNX 端口存在
  （aac6fef/laya-mlx、FluidInference/laya-coreml、itcast666/laya-vision-onnx）。
  **没有"基于 laya 的 locomotion 训练权重"**——laya 是决策层模型，
  与 PPO 续训不搭；续训正解是 tfrere model.pt 或官方 ONNX 做低层 +
  laya-vision 做高层（后续「VLM 驾驶」方向的现成积木）。

### 训练 LKG 直播链路 + venv 根治（2026-10-07）

用户要求：纯 Mac 端训练（弃 CUDA），且每次回归训练都要有 LKG 实机显示。

- **直播链路（端到端已验证）**：microduck_rl 新增 `lkg_stream.py`
  （`LKGStreamEnvProxy`，纯观测、异常全吞，`MICRODUCK_LKG_STREAM=<port>` 开启，
  挂在 MicroduckOnPolicyRunner 里 super().__init__ 之后包装 self.env）
  → UDP（127.0.0.1:44123，[magic,nq,sim_t,qpos]）→ viewer `--stream [port]` 模式
  （不跑本地 mj_step，逐包写 qpos+time → mj_forward → 正常 quilt 光栅上屏）。
  60 iters 训练全程 LKG 直播，reward 0.13→0.67，iter 耗时无变化（2.1s）。
- **相机跟随（踩坑）**：mjlab env origins 是网格布点，**env 0 在 (6.7,-6.8)**——
  相机对原点取景时鸭子在 7m 外（"LKG 上看不到鸭子"的根因）。
  修复：流模式相机中心跟随 base xy，地面 plane 跟随并放大到 64m 覆盖 env 网格。
- **venv 根治**：~/Documents 在 iCloud 同步域，fileproviderd 反复给 .venv 文件
  打 UF_HIDDEN（uv CPython 跳过 hidden .pth；uv run 每次 sync 还重建 bin/ 链接，
  并发 uv run 时 train shim/python 链接瞬缺）——`uv run train` 时好时坏的根源。
  **venv 迁到 `~/kimi-work/venvs/microduck_rl`（UV_PROJECT_ENVIRONMENT）后连续
  3× `uv run train` 全稳**。面板 TrainingRunner 已注入该 env var +
  MICRODUCK_LKG_STREAM（面板「LKG 直播」开关默认开）。
- **HF 现成参数（2026-10-07 调查）**：官方 `pollen-robotics/microduck-policies`
  10 个 ONNX（alpha_walking/stand/sitstand/ground_pick/ball_kick×2/roller/
  roller_crouch/roulade/velstand + manifest.json）；社区策略几十个
  （hf search "microduck"）；**`tfrere/microduck-move-base-walk` /
  `-ground-pick` 带 base/model.pt 检查点**（上游 recipe 复训 2999/5999 iters
  @4096，可作续训起点）。本地已有 microduck_m1_800.onnx / m1_5_2000.onnx。
  本机直连 huggingface.co 不通，**hf-mirror.com 正常**（API/下载走它）。
  wandb 在线 checkpoint 需 `wandb login`（本机未配）。

### 本地 CPU 训练打通 + MLX 迁移评估（2026-10-07，结论：不迁移）

背景：用户要求评估训练能否迁 MLX。实测结论先行：**不迁——本地训练不需要 MLX，
warp-CPU + torch-CPU 原栈直接能跑；物理采集占 96.6%，PPO 学习仅占 3.4%，
MLX 无论替哪层都没有切入点**。

- **本机训练链路（原栈，零迁移）**：`uv run train <task> --gpu-ids None`
  （mjlab 原生 CPU 模式，`select_gpus(None)`→device=cpu）+ `WANDB_MODE=offline`
  （本机无 wandb key）。64 envs×5 iters smoke 全绿：奖励项/课程/终止指标全出。
  面板「本地训练」区已固化：设备 CPU/CUDA 分段开关 + wandb 离线开关，
  面板发起→跑完→reward 曲线端到端验证（reward 0.130 ×5）。
- **速度标定**（M5 Max，velocity 任务，含物理+obs/reward+PPO 全链路）：
  64 envs 2.17s/iter（708 env-steps/s）· 256 envs 7.0s/iter（878/s）·
  1024 envs 27.0s/iter（910/s）——**吞吐 ~900 env-steps/s 即饱和**。
  分解（64 envs）：Collection 2.115s vs Learning 0.074s → 96.6% 物理。
- **够不够用**：smoke/课程调试/小实验（≤512 envs × 几百 iters，分钟~小时级）✔ 可用；
  简单 episodic trick（1000 iters @4096）≈ 30h 勉强；正式 gait run
  （4000–6000 iters @4096 ≈ 393–590M steps）≈ 5–7.5 天 ✘ 不现实。
  **正式训练仍是 CUDA/HF Jobs（技术栈分工不变），本地 CPU 是调试档。**
- **MLX 为何不迁**：① 物理：本机 warp devices=['cpu']，warp 无 Metal 后端；
  MLX 自研批量求解器早已否决（迭代求解器语义 + sim2real 红线 + 月级）。
  ② 学习：MLP 512×256×128 的 PPO update 在 torch-CPU 上 0.074s/iter（3.4%），
  换 MLX 省不到 3% 还要重写 rsl_rl + normalizer + rollout storage，
  并引入 warp-tensor↔mlx 拷贝边界。③ torch MPS 在本机可用，同样用不上
  （学习不是瓶颈）。
- 面板训练循环期间 LKG 显示循环全程 60.0 FPS 不掉帧（M5 Max 算力有余）。

### P2 ✅（2026-10-07 验收通过；实机目检以用户描述为准）

产物：viewer 新增 app 模式（无 --dump/--bench/--dump-lentic 时进入 LKGApp 壳）+
`--dump-lentic/--from-hdr/--test-pattern` 离线/半离线模式；StreamQuilt 侧增量：
预览窗 tile 网格（tonemap shader 可选 LKGGridParams，默认关→dump 输出逐位不变）、
LKGApp `onDidFinishLaunching` 钩子 / `setPaused()` / `showTileGrid`（g 键）、
`Calibration.fetchFromBridge` 默认改 `http://127.0.0.1:33334`（localhost 别名不可靠，
roadmap 既有结论落到源码默认值）。校准/quilt 常量零副本，全部源码级复用。

- **离线回归（验收 1）**：固定 quilt 输入（`sq-demo -- --dump-hdr /tmp/p2_ref.hdr`，
  saveQuiltHDR rgba16Float raw + JSON sidecar）→ viewer `--dump-lentic --from-hdr`
  输出 vs `sq-demo --dump-lentic`：**md5 完全一致**（3e0a511e01f08f924ebb1587b2b77a56，
  1440×2560），远严于 ±1。两边校准同来源（Bridge 在线，真机 serial LKG-E10707）。
  StreamQuilt 参考 quilt 回归不变：md5=caaf1d42f528a58ecd3eeaede99aa554 ✓。
- **上屏链路**：app 模式 = LKGApp（校准 fetch→lenticularUniforms→encodeLenticular→
  设备窗 60Hz）+ onRenderQuilt 里 mj_step×1 + mjv_updateScene + encodeScene。
  LKG 屏识别正常（"LKG-E10707 1440x2560 — live"）；LKG 窗 borderless
  orderFrontRegardless，永不抢焦点；交织图截屏目检条纹正常、鸭子居中。
- **校准测试图**（验收 2）：encodeTestPattern 经面板开关或预览窗 c 键触发
  （整屏纯色/转头色相连续属光学目检项，用户实机可用，未单独逐项回报；
  整体画面构图经用户确认"效果可以"——fov 35° 偏扁 → **45° 定为新默认**）。
- **主屏预览窗**（验收 3）：tonemapped 裸 quilt + tile 网格（g 键/面板开关，
  网格线宽 ~1 目标像素，fwidth 自适应缩放）。
- **实机帧率（验收 4，含 vsync/上屏开销，LKGApp FrameDriver 实测）**：
  11×6 全分辨率 4092²：**稳定 60.0 FPS**，GPU scene 8.4ms + interlace post 1.7ms
  ≈ 10.1ms/帧（预算 16.6ms）。**档位定案：11×6 全分辨率，不降档**（P1 顶点瓶颈
  分析仍成立；半分辨率/7×8 留作弱机降级，面板里可切）。
- **物理同进程（验收 5）**：app 循环每设备帧 mj_step×1，遥测 |qpos| 1.037→1.009
  （zero-ctrl 鸭子倾倒落地）、sim t 持续推进，面板与 stdout 双通道可见。

**P2 顺带先行落地的 P3 内容（用户现场要求）**：SwiftUI 控制面板（主屏独立窗，
NSHostingView 挂进 LKGApp 壳，经新增 onDidFinishLaunching 钩子）——
遥测（FPS/|qpos|/sim t/draws/tris）、相机（fov/方位角/距离倍率/目标偏移 XY，
实时重建取景，AX 驱动验证 fov 45→54 dist 0.614→0.499 联动正确；**不做俯仰**，
P1 定案）、显示（测试图/bypass/网格/暂停/分辨率/存 PNG）、物理（暂停/每帧步数/
重置 keyframe）、本地训练（uv run train 子进程启停 + 日志尾 + Mean reward 曲线）。

**P2 排掉的雷（环境事实，勿重复调研）**：microduck_rl 在 iCloud Desktop &
Documents 同步域内，fileproviderd 给 .venv 大量文件打 **UF_HIDDEN**（0x8040），
uv 版 CPython 的 site.py **跳过 hidden .pth** → editable 的 src/ 链接失效，
`uv run train` 裸跑 ModuleNotFoundError: mjlab_microduck（终端裸跑同病，非面板 bug）。
临时修复：`chflags nohidden`（会被 fileproviderd 重新打上）；面板训练启动器
统一注入 `PYTHONPATH=<repo>/src` 绕过（已验证）。**根治建议：RL 仓库迁出
~/Documents iCloud 同步域**（wandb/logs 也不该被同步），待用户决定。

### P1 ✅（2026-10-06 验收通过）

产物：新 executable target `mujoco-lkg-viewer`（`Sources/mujoco-lkg-viewer/main.swift` +
`MjvQuiltRenderer.swift`）。StreamQuilt 以 **path dependency**（`~/Desktop/StreamQuilt`
library product）接入，QuiltMath/QuiltSpec/QuiltRenderer/Calibration 零拷贝复用，
本 package 未另抄任何 quilt/校准常量。

用法：`swift run mujoco-lkg-viewer [scene.xml] [--layout 11x6|7x8] [--dump out.png]
[--bench N] [--steps N] [--half] [--sweep X --dist X --fov DEG --az DEG]`。

- mesh 上传：`mjModel.mesh_*` → 每 mesh 双 MTLBuffer（smooth=索引化+顶点法线 /
  flat=展开+面法线，按 scene geom `dataid&1` 选择，`dataid/2`=meshid）。
  **勘误（用户目检发现部件散架后定位）：mesh_vert/mesh_normal 在 mjC 编译期
  已烘焙 mesh_scale/mesh_pos/mesh_quat（头文件注释 "applied to asset vertices"），
  顶点已在 geom frame，上传后直接用、禁止再乘 mesh frame**——二次变换导致
  部件按 mesh frame 偏移/旋转（鸭子散架）。`--diag` 验证：mjvScene pose 与
  d.geom_xpos/xmat 完全一致（|Δ|=0），mesh frame 显著非恒等。
  真值对照：`microduck_rl/.venv` Python 离屏渲染（Renderer + keyframe 0，
  /tmp/mj_ref.png）。mesh_normal 缺失时按面累积重算。纹理未接（P1 材质简化）。
  相机默认 az=205°（正面 3/4，看到鸭脸黄眼圈）。
- 光栅器：单 render pass 覆盖 quilt（rgba16Float + depth32Float），逐 tile
  `setViewport(QuiltRenderer.tileRect)` + scissor；`QuiltCamera` 离轴投影，
  **相机水平无俯仰**（QuiltCamera 的剪切矩阵只消水平 offset；俯仰会让焦点平面
  跨视角像素失配 ~½(camH/dist)² 量级）。世界系变换 W = Rx(−90°)·Rz(az)·T(−bbox中心)
  （mujoco z-up → 相机 y-up，主体居中到焦点平面 z=0）。
- 材质/光照：albedo=mjvGeom.rgba（^2.2 转线性）+ 双方向光 + specular/shininess/
  emission（取自 mjvGeom），HDR 线性输出走 QuiltRenderer ACES tonemap，
  与参考 quilt 同管线同风格。
- 场景同步：每帧 `mjv_updateScene` → 遍历 71 scene geom（70 mesh + 1 plane）→
  instance uniforms（model/rgba/misc）。地面 quad 渲染半径放大到 max(XML size,
  3×dist)，否则前景露底（microduck floor size 只有 0.5×0.5）。
- 自动取景：非平面 geom 的 rbound bbox → 分轴贴合 dist（水平 half.x/竖直 half.z，
  tileAspect 参与）×1.32 边距，sweep=max(2·half.x, 0.8·r)，az 默认 25°（3/4 视角）。
- 验收数据（Apple Silicon，release 构建，200 帧均值，每帧 mj_step×1）：
  - **11×6=66 视角 4092² 全分辨率：wall 6.55ms（p50 6.30），GPU 5.62ms → ~152 FPS**
  - **7×8=56 视角 2016×4096：wall 5.15ms，GPU 4.38ms → ~194 FPS**
  - 11×6 半分辨率：wall 5.49ms（GPU 4.66ms）——顶点瓶颈（52.6M tris/帧，
    431k tris/视角），半分辨率收益仅 ~16%
  - 布局定案：**主选 11×6 全分辨率**（6.55ms ≪ 16.6ms 预算），无需半分辨率；
    7×8 留作弱机降级档。若未来场景三角面暴涨，先减 mesh LOD 再考虑半分辨率。
  - 焦点平面 z=0 像素级验证：跨视角最大偏差 **0.00003 px**（level camera 下
    QuiltCamera 剪切矩阵精确消 offset）。
  - 物理同进程：bench 200 帧 mj_step 持续推进，|qpos| 1.0072→1.0217，sim t 推进 0.4s。
  - dump：`/tmp/mduck_qs11x6a0.56.png`（4092²，keyframe 站立姿态立于地面，
    逐 tile 视差连续无跳变/镜像）；参考 quilt `/tmp/ref_qs11x6a0.56.png`
    本机复现 md5=caaf1d42f528a58ecd3eeaede99aa554 ✓。
- P1 坑：① **mesh_scale/pos/quat 已烘焙进顶点，二次应用 = 部件散架**（见勘误）；
  ② Swift 6 下顶层全局变量 main-actor 隔离，访问它们的全局函数要标 `@MainActor`；
  ③ 地面平面 XML size 可能很小，渲染半径须按相机距离下限放大；
  ④ `m.mesh_scale[i]` 是标量指针取值，不是数组——须 `m.mesh_scale[i*3+k]`。

### P0 ✅（2026-10-06 验收通过）

产物：本目录 SPM package（`Package.swift` + `Sources/mujoco-lkg-probe/` +
`Vendor/MuJoCo.xcframework`）。`swift run mujoco-lkg-probe [scene.xml] [--steps N]`。

- 链接方案：venv `libmujoco.3.10.0.dylib`（universal）+ `include/` → 拷贝副本
  `install_name_tool -id @rpath/libmujoco.3.10.0.dylib`（原 install name 指向
  `mujoco.framework/...`，不改 id 无法直接 rpath）→ 手写 module.modulemap
  （umbrella = mujoco/mujoco.h，剔除 C++ only 的 experimental/render 子目录）→
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild
  -create-xcframework`（本机 xcode-select 指 CLT，须带 DEVELOPER_DIR）。
  SPM 会拷贝 dylib 到可执行文件旁但不加 rpath，executable target 需
  `linkerSettings: unsafeFlags(["-Xlinker","-rpath","-Xlinker","@executable_path"])`。
- 验收数据（microduck scene.xml）：nq=21 nv=20 nu=14 nbody=16 njnt=15 ngeom=82
  nmesh=38 nmat=39 ntex=2 nkey=4；15 关节/16 body 名称与 Python 侧
  `mj_id2name` 逐一相符；mj_step×1000 mean=0.015ms（预算 ~0.1ms 内）；
  zero-ctrl 自由落体 |Δqpos|=0.608，动力学活跃。
- mjvScene 走查：`mjv_makeScene` 只分配，**必须再 `mjv_updateScene`**
  （P0 踩坑：缺它 ngeom=0；cam 传 NULL 会 mju_error abort，须
  `mjv_defaultFreeCamera`）。结果 71 个 scene geom（70 mesh + 1 plane）。
- **P1 关键事实：scene geom 的 dataid = 2×meshid（偶=smooth/奇=flat），
  光栅器取 mesh 时须 ÷2**；objtype=5(mjOBJ_GEOM)、objid=模型 geom id。
