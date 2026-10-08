推进 MuJoCo-LKG P3：实时 sim 链路 + 视窗/控件分离。

项目：~/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg
先读 roadmap.md 的「进度日志」确认 P0–P2 已验收通过，并核对 P2 留下的产物形态
（app 模式接线、窗口结构、降级档位），然后只做 P3，验收通过才停。
主线仍是 Swift + Metal 单进程原生 app。实机目检由用户执行。

P0–P2 已固化事实（勿重复调研，细节以 roadmap.md 进度日志为准）：
- `mujoco-lkg-viewer` = P1 光栅器 + P2 上屏：MjvQuiltRenderer（quilt 逐 tile 离轴
  光栅，相机水平无俯仰）+ StreamQuilt 显示链（Calibration.fetchFromBridge →
  encodeLenticular → LKGDeviceWindowController 60Hz）+ 主屏预览窗。
  CLI 模式（--dump/--bench/--diag）与 app 模式并存。
- 物理：mj_step mean 0.015ms，不是瓶颈。microduck timestep=0.002s，
  策略步 50Hz（具体 substep 数与指令语义以 infer_policy.py 为准，禁止拍脑袋）。
- 性能基线（release）：11×6 全分辨率 wall ~6.6ms/帧 → ~150 FPS；
  瓶颈在顶点吞吐（52.6M tris/帧），半分辨率收益仅 ~16%。
- mesh 顶点已在 geom frame（mesh frame 编译期烘焙，禁止再乘）。
- quilt/校准/interlace 常量全部复用 StreamQuilt，本 package 零副本。

P3 任务（roadmap.md 原文）：
1. sim loop 与渲染解耦：mj_step（50Hz 策略步）与 quilt 渲染各自节拍，
   渲染跟不上时隔帧轮换视角子集（viewport 分级更新）。
2. SwiftUI 控制面板（主屏）：速度指令滑杆/按键、遥测显示；
   键盘事件走主屏窗口，LKG 窗永不抢焦点。
3. 策略推理接入：ONNX → CoreML（ANE 推理侧）或 onnxruntime CoreML EP；
   指令语义与 infer_policy.py 的 pty 键盘方案对齐（rehearsal_mcp 兼容）。
4. 性能打磨：掉帧策略（先降 interlace 输入分辨率，再降视角数）。

技术储备直接用，勿重复调研：
- 策略模型：microduck_rl 的 microduck_m1_800.onnx；推理侧语义（obs 构造、
  action 缩放、默认姿态、command 来源）以 microduck_rl/scripts/infer_policy.py
  为唯一基准，逐行对齐，不许自创。microduck_rl 侧改动遵守其 AGENTS.md
  （sim2real 不变量是红线）。
- 指令按键语义参考 rehearsal_mcp（rehearsal_key：up/down/left/right/space/t/p/q/
  h/b 等）——那是 pty 注入 infer_policy.py 的键位；本 app 主屏按键要产生
  同样的指令效果。
- StreamQuilt LKGApp 壳已有 onKey/onStatusLine 钩子（sq-demo 用法参考）；
  SwiftUI 控制面板可挂在主屏预览窗同 app 内。
- CoreML 转换：coremltools（microduck_rl venv 里若无则单独装，勿污染训练 venv）
  或 onnxruntime CoreML EP；先离线验证 CoreML 输出与 ONNX 参考输出逐元素一致
  （atol 1e-4），再进实时回路。
- 遥测：mcp__microduck-rehearsal 的遥测格式（achieved/cmd 速度、trunk_z）可作
  控制面板显示字段参考。

验收（全过才在 roadmap.md 进度日志记 P3 ✅ 并停下汇报）：
1. microduck 行走彩排实机 60FPS（或明确记录的稳定降级档位：降了什么、
   稳定多少 FPS、为何选此档）。
2. 主屏控制（速度指令/策略开关）LKG 光场实时跟随，鸭子行走、转向、停下
   响应正确；转头视差连续无跳变（用户目检）。
3. 策略推理输出与 infer_policy.py 同指令同初始状态下的参考轨迹一致
   （离线对拍：固定 N 步，关节角误差有数据）。
4. 键盘事件只走主屏窗口；LKG 窗全程不抢焦点（用户确认）。

纪律：校准/quilt 常量一律复用 StreamQuilt；策略 obs/action 语义以
infer_policy.py 为基准禁止自创；改相机/interlace 数学后用固定输入离线回归；
实机目检结果以用户描述为准记录进 roadmap。
