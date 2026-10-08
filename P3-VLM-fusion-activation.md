# P3 融合方案激活提示词：RL 低层策略 × laya-vision 高层驾驶 × LKG 光场

> 用法：新 session 里粘贴「激活提示词」整段；先把「模型下载命令」跑完。

## 模型下载命令（本机直连 HF 不通，一律走 hf-mirror）

```bash
# 下载工具：用训练 venv 自带的 hf CLI（huggingface_hub 已是依赖）
HF=/Users/jup33q/kimi-work/venvs/microduck_rl/bin/hf
export HF_ENDPOINT=https://hf-mirror.com
M=/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg/models
mkdir -p "$M"

# 1) 官方部署级策略 ×10（alpha_walking/stand/sitstand/ground_pick/ball_kick×2/
#    roller/roller_crouch/roulade/velstand + manifest.json）
$HF download pollen-robotics/microduck-policies --local-dir "$M/pollen-policies"

# 2) 续训检查点：walk 2999 iters @4096（base/model.pt + agent.yaml + env.yaml）
$HF download tfrere/microduck-move-base-walk --local-dir "$M/tfrere-base-walk"

# 3) VLM 高层决策：laya-vision microduck-kick 微调（201M，safetensors + processor）
$HF download thaitea/laya-vision-microduck-kick --local-dir "$M/laya-vision-microduck-kick"

# 4)（备选）laya-vision 基座 / ONNX 端口（若要自己微调或走 onnxruntime）
# $HF download thaitea/laya-vision --local-dir "$M/laya-vision-base"
# $HF download itcast666/laya-vision-onnx --local-dir "$M/laya-vision-onnx"

# 本地已有：microduck_rl/microduck_m1_800.onnx、microduck_m1_5_2000.onnx（走路）
```

续训用 tfrere 检查点（放到 logs 下让 mjlab resume 语义找到它）：

```bash
RL=/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/microduck_rl
mkdir -p "$RL/logs/rsl_rl/velocity/tfrere-base"
cp "$M/tfrere-base-walk/base/model.pt" "$RL/logs/rsl_rl/velocity/tfrere-base/"
# 之后训练加：--agent.resume True --agent.load-run tfrere-base
# （面板「额外参数」里填这段也行；自动续跑开启时首块加，后续块自动 resume 最新）
```

---

## 激活提示词（粘贴给新 session）

````text
推进 MuJoCo-LKG P3 融合版：RL 低层策略 + laya-vision VLM 高层驾驶，LKG Go 光场实机显示。

项目：~/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg
先读 roadmap.md 进度日志（P0/P1/P2 已验收；直播链路、CPU 训练、自动续跑均在）。
主线仍是 Swift + Metal 单进程原生 app（mujoco-lkg-viewer）。

已固化事实（勿重复调研）：
- viewer：swift run mujoco-lkg-viewer [--stream [port]] [--layout 7x8|11x6] [--dump/--bench/
  --dump-lentic/--from-hdr …]。默认 7x8=56 views、fov 45°。app 模式 = LKGApp 壳
  （LKG 设备窗 60Hz + 主屏预览窗 + SwiftUI 控制面板）。控制面板在
  Sources/mujoco-lkg-viewer/ControlPanel.swift（遥测/相机/显示/物理/本地训练五区）。
- 训练直播：microduck_rl/src/mjlab_microduck/lkg_stream.py（MICRODUCK_LKG_STREAM=44123
  开 UDP qpos 推流，纯观测）；viewer --stream 接收渲染，相机/地面跟随 env 0。
  面板训练启动器：CPU 档（--gpu-ids None）+ wandb offline + 20 iter/块自动 resume 续跑。
- 训练 venv 已迁出 iCloud：UV_PROJECT_ENVIRONMENT=~/kimi-work/venvs/microduck_rl
  （~/Documents 在 iCloud 同步域，fileproviderd 会给 .venv 打 UF_HIDDEN，
  uv 版 CPython 跳过 hidden .pth——不要再把 venv 放回去）。
- 模型已下载到 mujoco-lkg/models/：pollen-policies（官方 10×ONNX）、
  tfrere-base-walk（model.pt 续训检查点）、laya-vision-microduck-kick（VLM）。
- 本机 HF 直连不通，下载走 HF_ENDPOINT=https://hf-mirror.com。
- StreamQuilt（~/Desktop/StreamQuilt，path dependency）校准/quilt 常量一律源码复用，
  禁止另抄；改 shader/相机数学后用 --dump-lentic --from-hdr 固定输入回归
  （参考：cd ~/Desktop/StreamQuilt && swift run -c release sq-demo -- --dump-hdr …）。
- microduck_rl 改动遵守其 AGENTS.md：61D obs 布局（48 proprio + 13D 命令块
  [twist(3), head_pose(4), body_pose(6)]）、关节序号、BAM 执行器、无滤波等是
  sim2real 红线；策略导出走 scripts/export.py（normalizer 烘焙进 ONNX）。

目标架构（融合两方案）：
1. 低层 RL 策略接入 viewer（本地推理，替代/叠加现有 mj_step 自由演化和训练直播）：
   ONNX Runtime（venv 已有 onnxruntime；Swift 侧可用 onnxruntime C API 或
   子进程推理 + UDP 动作回传，二选一，优先同进程）。obs 组装严格对齐
   microduck_rl/scripts/infer_policy.py 的 61D 语义（含 BAM 执行器状态、
   50Hz 策略节拍、命令块写入约定——posture flag 在 twist vx 槽位，读源码对齐）。
   起步用 microduck_m1_800.onnx（走路）；策略热切换挂 pollen-policies 的
   stand/sitstand/ground_pick/ball_kick 等（manifest.json 有命令契约）。
2. 高层 VLM 驾驶：viewer 渲一路相机帧（鸭子头部相机或第三人称 256×256）
   → laya-vision-microduck-kick（201M，四选一 FORWARD/LEFT/RIGHT/KICK）
   → 映射成低层策略命令（twist vx/yaw；KICK 切 ball_kick 策略或触发对应
   命令块）。VLM 推理频次 1-2Hz 足够。Mac 原生推理：模型是 safetensors
   （SmolVLM 类），可用 MLX Swift（mlx-swift-examples 有 SmolVLM 参考）或先
   Python 子进程（venv + transformers）+ UDP 回传跑通再议移植。
   注意它只在 quackd 两个模拟器上验证过（研究性质），画面域差是大坑，
   先用 bypass/预览窗对比确认它认得出我们渲染的鸭子再接控制环。
3. 融合展示：LKG 上鸭子自主行动（VLM 决策 + RL 执行），主屏面板显示
   当前策略/VLM 决策/置信度；训练直播链路保持可用（--stream 与本地策略
   模式互斥切换即可）。

验收：
1. viewer 本地加载 m1_800.onnx，鸭子在 LKG 上行走（面板速度指令可调），
   obs 语义与 infer_policy.py 逐字段对齐（打表对照）。
2. VLM 决策环闭合：渲染帧 → laya-vision → 四选一 → 低层执行，LKG 实机可见
   鸭子朝球走/踢球（无球场景则 FORWARD/转向巡航）。VLM 误判要有日志。
3. 面板可切换：本地策略驾驶 / VLM 融合 / 训练直播（--stream）。
4. 60 FPS 不降档（56v 全分辨率基线 scene ~7.8ms + post ~2.4ms，VLM 推理
   不得阻塞渲染线程——放后台线程/子进程）。

纪律：校准/quilt 常量复用 StreamQuilt；LKG 窗永不抢主屏焦点；
microduck_rl 只加观测性代码；实机目检结果以用户描述为准记进 roadmap。
````

## 融合要点备忘（为什么这样拼）

- laya-vision-microduck-kick 是**高层决策**模型（四选一离散动作），不能替代
  PPO 策略；它的低层行走在 quackd 里也是调用上游训练策略——所以我们同样
  VLM 管「去哪/做什么」，RL 策略管「怎么走」。
- 「基于已有权重训练」走 tfrere model.pt resume（2999 iters 起步，省一半预算）；
  与 VLM 融合是两条正交轴，可独立推进。
- 纯 Mac 推理：VLM 201M 在 M5 Max 上 MLX/ONNX CPU 都够 1-2Hz 决策节拍。
