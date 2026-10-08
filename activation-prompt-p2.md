推进 MuJoCo-LKG P2：interlace 上屏 + 校准通道。

项目：~/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg
先读 roadmap.md 的「进度日志」确认 P0/P1 已验收通过，然后只做 P2，验收通过才停。
主线仍是 Swift + Metal 单进程原生 app；本阶段是把 P1 的 quilt 渲染接上
StreamQuilt 显示侧（Calibration → encodeLenticular → LKGDeviceWindowController），
实机目检由用户执行，你负责出离线比对数据和上屏窗口。

显示路线已定案：**真 quilt 路线（自研光栅 + 自研 interlace 窗口上屏），不做
Bridge RGBD**。RGBD 路径已评估否决（DIBR 单视角外推在剪影边缘有遮挡空洞，
质量劣于真 quilt；且同样要求接线，非无线投屏）。分辨率/布局允许降档：
11×6=66 或 7×8=56 都行，全分辨率或半分辨率都行——以实机稳定 60FPS 为准，
选定档位后把帧耗时数据和降档理由记进 roadmap。

P0/P1 已固化事实（勿重复调研）：
- SPM package：`mujoco-lkg-probe`（P0 诊断）+ `mujoco-lkg-viewer`（P1 光栅器）。
  StreamQuilt 以 path dependency（~/Desktop/StreamQuilt library product）接入，
  QuiltMath/QuiltSpec/QuiltRenderer/Calibration 源码级复用，本 package 无常量副本。
- viewer 用法：`swift run mujoco-lkg-viewer [scene.xml] [--layout 11x6|7x8]
  [--dump out.png] [--bench N] [--steps N] [--half] [--sweep X --dist X --fov DEG
  --az DEG] [--diag]`。当前是 CLI dump/bench 形态，无窗口循环。
- P1 光栅器：MjvQuiltRenderer（单 render pass 覆盖 quilt rgba16Float + depth，
  逐 tile MTLViewport + QuiltCamera 离轴投影，相机水平无俯仰——剪切矩阵只消
  水平 offset）。encodeScene(cmd:items:) 可直接挂进 LKGApp.onRenderQuilt。
  collectDrawItems() 每帧从 mjvScene 同步 geom 位姿（mj_step 在帧循环里跑）。
- mesh 上传：mesh_vert/mesh_normal 编译期已烘焙 mesh frame，直接用，
  禁止再乘 mesh_pos/quat/scale（P1 踩坑，二次应用会散架）。
- 性能基线（release，200 帧纯渲染，不含显示循环/vsync 开销）：
  11×6 全分辨率 wall 6.55ms；11×6 半分辨率 5.49ms；7×8 全分辨率 5.15ms。
  瓶颈在顶点吞吐（52.6M tris/帧），半分辨率收益有限；优先 11×6 全分辨率，
  显示循环实测不达标再按 --half → 7×8 顺序降档。
  焦点平面 z=0 跨视角偏差 0.00003px。
- 参考 dump：/tmp/mduck_qs11x6a0.56.png（az=205° 正面 3/4 站立鸭子）；
  StreamQuilt 参考 /tmp/ref_qs11x6a0.56.png（md5 caaf1d42f528a58ecd3eeaede99aa554）。

P2 任务（roadmap.md 原文）：
1. 接入 StreamQuilt 显示侧：Calibration.fetchFromBridge（失败回退 lkgGoFallback）
   → lenticularUniforms → QuiltRenderer.encodeLenticular
   → LKGDeviceWindowController.show()。
2. 校准测试图模式（encodeTestPattern 现成）：整屏纯色、转头色相连续。
3. 主屏预览窗：tonemapped 裸 quilt + tile 网格。

技术储备直接用，勿重复调研：
- StreamQuilt LKGApp（Sources/StreamQuilt/LKGApp.swift）是现成 app 壳：
  `let app = try LKGApp(spec: .lkgGo, renderScale:)` → `app.onRenderQuilt =
  { cmd, pass, time in ... }` → `app.run()`，自带主屏预览窗与 LKG 设备窗
  60Hz 推流（参考用法：~/Desktop/StreamQuilt/Sources/sq-demo/main.swift）。
  建议给 viewer 加 app 模式：无 --dump/--bench 时进 LKGApp 循环，
  onRenderQuilt 里 mj_step + mjv_updateScene + encodeScene。
- Bridge REST 是 **PUT** `127.0.0.1:33334`（不要 localhost）。Bridge 不在线时
  回退 lkgGoFallback 并打印警告（实机目检需要 Bridge 在线拿真机校准）。
- interlace 数学已逐像素验证，直接用 LKGFixedShaders / Calibration，
  禁止另抄 pitch/tilt/subp 常量。
- 离线回归：interlace 输出 vs `cd ~/Desktop/StreamQuilt && swift run -c release
  sq-demo -- --dump-lentic <path>`，同输入 quilt 比对 ±1（maxOut=0.7 钳制必须一致；
  ~/lkg-viewer-godot 有 Python oracle 脚本可借）。
- ~/lkg-viewer-godot/docs/lkg-roadmap.md 有上屏踩坑清单可参考。

验收（全过才在 roadmap.md 进度日志记 P2 ✅ 并停下汇报）：
1. 离线：同一 quilt 输入经本 viewer 的 encodeLenticular 输出与 streamquilt
   --dump-lentic 比对 ±1（固定输入回归，证明 interlace 接线无改动）。
2. 上屏（用户实机目检为准）：校准测试图整屏纯色、转头色相连续无彩虹纹；
   microduck quilt 上屏视差正确（转头时鸭子立体感、无内外反/重影，
   flip 方向不对就调 QuiltCamera.flip）。
3. 主屏预览窗正常显示 tonemapped 裸 quilt + tile 网格。
4. 实机显示循环帧率数据（含 vsync/上屏开销的 ms/frame 与 FPS），
   据此定最终布局档位（11×6 全分辨率 / 11×6 半分辨率 / 7×8），记进 roadmap。
5. 物理仍同进程推进（app 循环里 mj_step，qpos 随时间变化，遥测可打印）。

纪律：校准/quilt 常量一律复用 StreamQuilt 源码，禁止另抄一份；
改 shader/相机数学后用固定 quilt 输入做离线回归；LKG 窗永不抢主屏焦点；
实机目检结果以用户描述为准记录进 roadmap。
