# MuJoCo-LKG 激活提示词（整合版，2026-10-07）

> 用法：新 session 整段粘贴「激活提示词」。前置：先跑「模型下载命令」（若
> mujoco-lkg/models/ 已有内容则跳过）。本文档同时是项目当前的整合 plan。

## 模型下载命令（官方 hf CLI 直连；本机 fake-ip 已修，HF 直连正常）

**2026-10-07 已全部下载并验证**（pollen-policies 13 文件 / tfrere-base-walk 含
base/model.pt（actor+critic+optimizer state dict，iter 续训用）/
laya-vision-microduck-kick 808MB safetensors）。以下命令留作重装/换机用：

```bash
HF=/Users/jup33q/kimi-work/venvs/microduck_rl/bin/hf   # venv 自带 huggingface_hub CLI
M=~/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg/models
mkdir -p "$M"

# 官方部署级策略 ×10（walking/stand/sitstand/ground_pick/ball_kick×2/roller/roller_crouch/roulade/velstand）
$HF download pollen-robotics/microduck-policies --local-dir "$M/pollen-policies"

# 续训检查点（walk 2999 iters @4096，含 agent/env yaml）
$HF download tfrere/microduck-move-base-walk --local-dir "$M/tfrere-base-walk"

# VLM（laya-vision microduck-kick 微调，201M safetensors）
$HF download thaitea/laya-vision-microduck-kick --local-dir "$M/laya-vision-microduck-kick"
```

续训接入：`cp $M/tfrere-base-walk/base/model.pt →
microduck_rl/logs/rsl_rl/velocity/tfrere-base/`，训练加
`--agent.resume True --agent.load-run tfrere-base`。

（若哪天 HF 直连又挂——ClashX 订阅更新覆盖 fake-ip-filter 的症状——把
`*.huggingface.co` 加回 ~/.config/clash.meta/Yecao100-*.yaml 的
dns.fake-ip-filter，或临时 `export HF_ENDPOINT=https://hf-mirror.com`。）

---

## 激活提示词（粘贴给新 session）

````text
继续 MuJoCo-LKG 项目。先读 ~/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/
mujoco-lkg/roadmap.md 进度日志（P0/P1/P2 已验收，P3 进行中）。

## 系统现状（勿重复调研）

主线：Swift + Metal 单进程 app `mujoco-lkg-viewer`（SPM，StreamQuilt path
dependency 在 ~/Desktop/StreamQuilt，校准/quilt 常量源码复用禁止另抄）。
- 显示：7x8=56 views 默认、fov 45°、60.0 FPS；LKG 设备窗 + 主屏预览窗
  （tonemapped quilt + tile 网格）+ SwiftUI 控制面板。
- 策略驾驶（P3 已落地）：PolicyRunner.swift = onnxruntime C API（C shim target
  Sources/COnnxRuntime，dylib 在 Vendor/OnnxRuntime；xcframework 路线被 SPM
  modulemap 撞名否决）本地推理，obs 61D 逐字段对齐 infer_policy.py（训练契约
  细节在 roadmap「P3 起步」节）。执行器默认 **BAM M6**（BamController.swift，
  bit 级对照验证过；--no-bam 回退 XML PD）。
  CLI：--policy [onnx] [--ducks N] [--no-bam] [--bam-trace out.csv]；面板「策略驾驶」区：策略 picker / WASD+方向键
  （按住走松开停）/ 速度滑杆 / 推倒（空格）/ 鸭子数。多鸭子顶点吞吐 ×N
  （4 只 56v ≈ 47 FPS，>2 只降半分辨率）。
- 训练直播：microduck_rl/src/mjlab_microduck/lkg_stream.py（MICRODUCK_LKG_STREAM=
  44123 UDP qpos，纯观测）→ viewer --stream 接收 + 相机/地面跟随。
- 本地训练（纯 Mac，无 CUDA）：面板「本地训练」区，CPU 档（--gpu-ids None）+
  wandb offline + 20 iter/块自动 resume 续跑；吞吐 ~900 env-steps/s 单实例
  （物理采集占 96.6%，MLX 评估结论=不迁移，记录在技术栈分工）。

## 关键环境事实（踩过的雷）
- 训练 venv 已迁出 iCloud：UV_PROJECT_ENVIRONMENT=~/kimi-work/venvs/microduck_rl
  （~/Documents 的 .venv 会被 fileproviderd 打 UF_HIDDEN，uv CPython 跳过
  hidden .pth）。mjpython 在新 venv 需 libpython 软链（已建）。
- ClashX fake-ip 已加 fake-ip-filter（r2.cloudflarestorage / ollama / hf 系）；
  订阅更新会覆盖，症状复发（ollama pull 报 198.18.x）就重加。
- BAM 真值：alpha_walking BAM achieved +0.11 m/s（cmd 0.3）；m1_800 BAM/PD 都
  不走（里程碑策略，不是 PD 的锅）。XML PD 与 BAM 行为差异已对齐参考实现。

## MCP 接口
- microduck-rehearsal（~/.kimi-code/mcp.json → rehearsal_mcp/server.py，
  v0.2 多 worker）：rehearsal_start(id?, onnx, extra_args)/key(id,…)/telemetry/
  tail/screenshot(id)/list/stop。worker = mjpython GUI + pty 键盘注入；
  行走/轮滑 1:1 多开：roll 用 extra_args='--roller'。
- microduck-rehearsal 工具 schema 变更需重启 session 生效。

## 整合 plan（按优先级）
1. ~~BAM 执行器移植到 viewer~~ ✅ 2026-10-07 完成（BamController.swift，默认开，
   --no-bam 回退；bit 级对照：XML PD 全程 max|Δ|=0，BAM 前 1.25s ≤1.4e-16，
   achieved/cmd +0.11/+0.30 = 参考值。细节见 roadmap「BAM 执行器移植」节）。
2. VLM 高层驾驶（laya-vision-microduck-kick）：viewer 渲相机帧 256×256 →
   VLM 四选一（FORWARD/LEFT/RIGHT/KICK）→ 低层命令映射；1-2Hz 后台推理，
   不阻塞渲染。先确认它认得出我们渲染的鸭子（域差坑）。
3. 续训：tfrere 检查点 resume（命令见上），自动续跑串联。
4. 通用化收尾（roadmap P4）：任意 MJCF 5 分钟上屏。
5. 桌面封装（roadmap P5）：viewer 打包成 .app 放 ~/Desktop（embedded
   dylibs、路径解硬编码、双击启动恢复上次配置、LKG 不在线降级纯预览），
   全流程免终端。

纪律：microduck_rl 只加观测性代码（其 AGENTS.md 红线：61D obs 布局/BAM/
无滤波/DR 不累积）；StreamQuilt 常量复用；LKG 窗永不抢主屏焦点；
改 shader/相机数学后跑 --dump-lentic --from-hdr 固定输入回归；
实机目检以用户描述为准记进 roadmap。
````

## 当前会话遗产（后台任务）

- viewer 实例起停：`swift run -c release mujoco-lkg-viewer -- [--policy x.onnx]
  [--ducks N] [--stream] [--layout 7x8]`
- 训练：viewer 面板「开始训练」（自动续跑 20 iter/块）或手动
  `UV_PROJECT_ENVIRONMENT=~/kimi-work/venvs/microduck_rl PYTHONPATH=$PWD/src \
   WANDB_MODE=offline MICRODUCK_LKG_STREAM=44123 uv run train <TASK> \
   --env.scene.num-envs 64 --agent.max_iterations 20 --gpu-ids None`
- 彩排 worker：`rehearsal_start`（MCP，多开带 id）
