# MuJoCo-LKG 激活提示词

激活 MuJoCo-LKG 路线图。项目：~/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg
主线：Swift + Metal 单进程原生 app（libmujoco C API 物理 + mjvScene→Metal 光栅 +
StreamQuilt 显示管线 + SwiftUI 控制面板）；Python+OpenGL 只是备选快速路线。

先读固化 plan：mujoco-lkg/roadmap.md，确认当前阶段（从未做过则从 P0 开始），
按该阶段任务清单执行并跑验收，验收通过才进下一阶段。

技术储备直接用，勿重复调研：
- StreamQuilt（~/Desktop/StreamQuilt）：显示侧几乎全复用——
  Sources/StreamQuilt/ 的 QuiltRenderer（HDR quilt 目标/interlace/tonemap/测试图）、
  LKGDeviceWindowController（LKG 屏 60Hz 窗）、Calibration（Bridge REST + 回退值）、
  QuiltMath（离轴投影矩阵/离轴相机）、LKGApp（SwiftUI 壳 + 预览窗）
- libmujoco 来源已确认：microduck_rl/.venv/.../site-packages/mujoco/ 下
  libmujoco.3.10.0.dylib + include/ 头文件；brew 有 mujoco 3.15.0（升级备选）
- 已逐像素验证的 interlace 移植与踩坑归档：~/lkg-viewer-godot
  （lkg/lkg_interlace.gdshader、docs/lkg-roadmap.md）
- 参考 quilt 生成：cd ~/Desktop/StreamQuilt && swift run -c release sq-demo --
  --dump /tmp/ref_qs11x6a0.56.png（md5 caaf1d42f528a58ecd3eeaede99aa554）
- Bridge：必须 PUT，用 127.0.0.1 不用 localhost，hwid 含 "LKG"，
  windowCoords 是跨屏开窗的可靠坐标源
- LKG Go：1440×2560 竖屏，quilt 11×6=66 视角 4092² tile 372×682；备选 7×8=56

已定案的技术栈分工（勿再摇摆）：
MuJoCo C 库 = 物理（sim2real 真实性基准，不换不重写）；Metal = 渲染/显示；
MLX = 训练侧实验；ANE/CoreML = 策略推理；CUDA/HF Jobs = 正式训练。
已排除：MJX-on-jax-metal（缺算子不可用）、MuJoCo Warp/Newton（CUDA only）、
MLX/ANE 做物理、经 Godot 显示（无零拷贝 + 多屏坑）。

microduck_rl 侧有 AGENTS.md（sim2real 不变量是红线）：任何 infer_policy 相关
重构只加显示层，默认行为不变，跑 uv run --with pytest pytest tests/ 回归。

实机目检以 LKG Go 设备上转头视差为准，截图只是辅助。
