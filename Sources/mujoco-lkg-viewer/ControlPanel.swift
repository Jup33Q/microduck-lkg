import AppKit
import SwiftUI
import StreamQuilt

// P2/P3: SwiftUI 控制面板（主屏独立窗口）。LKGApp 壳仍负责 LKG 设备窗与预览窗，
// 面板经 NSHostingView 挂在主屏；显示开关直接写 LKGApp，相机/物理由渲染循环
// 每帧读取（cameraVersion / resetRequest 变更检测）。LKG 设备窗永不抢焦点。

@MainActor final class ViewerControl: ObservableObject {
    let app: LKGApp

    // ---- 相机（改动即 bump cameraVersion，渲染循环检测到后重建取景） ----
    @Published var fovDeg: Double { didSet { cameraVersion &+= 1 } }
    @Published var azDeg: Double { didSet { cameraVersion &+= 1 } }
    /// 距离倍率（×自动贴合距离；1.32 = P1 取景边距）
    @Published var margin: Double { didSet { cameraVersion &+= 1 } }
    /// 目标偏移（相机帧，×主体半径）：水平 / 竖直构图平移（相机无俯仰，见 P1 定案）
    @Published var offsetX: Double = 0 { didSet { cameraVersion &+= 1 } }
    @Published var offsetY: Double = 0 { didSet { cameraVersion &+= 1 } }
    private(set) var cameraVersion: UInt = 0

    // ---- 物理（渲染循环每帧读取） ----
    @Published var physicsPaused = false
    @Published var stepsPerFrame = 1
    @Published var resetRequest: UInt = 0

    // ---- 策略驾驶（P3；driveModeVersion 变更由渲染循环应用） ----
    enum DriveMode: String, CaseIterable {
        case free = "自由物理"
        case policy = "策略驾驶"
    }
    static var defaultPolicyPath: String { AppRes.defaultPolicy }
    @Published var driveMode: DriveMode { didSet { driveModeVersion &+= 1 } }
    /// 换策略只 bump policyVersion（渲染循环热切换，不复位鸭子）；
    /// 模式/BAM 切换才走 driveModeVersion（原地复位）
    @Published var policyPath: String { didSet { policyVersion &+= 1 } }
    /// BAM M6 执行器（训练同款电压模型）vs XML PD（--no-bam 语义）；切换重走 setupDucks
    @Published var actuatorBAM = true { didSet { driveModeVersion &+= 1 } }
    private(set) var driveModeVersion: UInt = 0
    private(set) var policyVersion: UInt = 0
    /// twist 速度指令（m/s, m/s, rad/s；infer_policy 键盘语义：前进 ±0.3、侧移 ±0.2、转 ±1.5）
    @Published var cmdVX: Double = 0
    @Published var cmdVY: Double = 0
    @Published var cmdYaw: Double = 0
    /// 随机推倒（infer_policy P 键语义：trunk xy 随机方向 1.0 m/s）
    @Published var pushRequest: UInt = 0
    /// 鸭子数量（多实例同屏，指令广播）
    @Published var duckCount = 1 { didSet { duckVersion &+= 1 } }
    private(set) var duckVersion: UInt = 0
    /// 相机平滑跟随 base xy（关 = 跳变跟随）；朝向跟随（相机遇鸭子转身缓慢摇到其身后）
    @Published var followSmooth = true
    @Published var followHeading = true

    // ---- VLM 高层驾驶（laya-vision-microduck-kick；竞技场场景限定） ----
    /// 场景具备 VLM 前置（head_camera + ball_free）
    var vlmArenaOK = false
    /// VLM 驾驶开关（开 = sidecar 决策覆盖 WASD 指令）
    @Published var vlmDrive = false
    /// 最近一次决策（choice + 四路概率 + 耗时）
    @Published var vlmStatus = "—"
    /// VLM 所见的头相机帧（256×256，决策时刻回读）
    @Published var vlmFrame: NSImage?

    // ---- 头部（颈）控制：13D 命令块 head_pose(4) = [neck_pitch, head_pitch, head_yaw,
    // head_roll]，home 位增量（quackd look() 同款：head_pitch 正 = 低头，head_yaw 正 = 左，
    // 限位 ±35°/±60°，上游 ±1.40 rad 内）。策略自己把头开到目标（实测有效，见 roadmap）。 
    @Published var headPitchDeg: Double = 0
    @Published var headYawDeg: Double = 0

    /// 候选策略：bundle/dev 解析的 pollen-policies 全量 + 开发仓库 m1 系列（P5 解硬编码）
    static var availablePolicies: [String] { AppRes.availablePolicies }

    // ---- 本地训练（microduck_rl，uv run train；runner 由视图持有） ----
    @Published var trainTask = "Mjlab-Velocity-Flat-MicroDuck"
    @Published var trainEnvs = 64
    @Published var trainIters = 100
    @Published var trainExtraArgs = ""
    /// true = --gpu-ids None（本机无 CUDA，warp-CPU + torch-CPU）；
    /// false = 默认 [0]（CUDA 机）。
    @Published var trainDeviceCPU = true
    /// 本机无 wandb API key，离线跑事后 sync。
    @Published var trainWandbOffline = true
    /// 训练时向 viewer 推 qpos 直播流（MICRODUCK_LKG_STREAM，端口 44123）。
    @Published var trainLkgStream = true
    /// 每块跑 trainIters 轮后自动 resume 最新 checkpoint 续跑，直到手动停。
    @Published var trainAutoLoop = true

    /// microduck_rl 仓库根（uv run 的工作目录）；P5：不存在则为 nil（面板隐藏训练区）
    var trainingWorkDir: String { AppRes.rlRepo ?? AppRes.devRlRepo }

    // ---- 遥测（渲染循环 ~0.25s 节奏写入） ----
    @Published var fps: Double = 0
    @Published var qpos: Double = 0
    @Published var simT: Double = 0
    @Published var draws: Int = 0
    @Published var tris: Int = 0
    @Published var distInfo: String = ""

    init(app: LKGApp, fovDeg: Double, azDeg: Double, margin: Double,
         driveMode: DriveMode = .free, policyPath: String = ViewerControl.defaultPolicyPath) {
        self.app = app
        self.fovDeg = fovDeg
        self.azDeg = azDeg
        self.margin = margin
        self.driveMode = driveMode
        self.policyPath = policyPath
    }

    // ---- 显示开关：直接镜像 LKGApp ----
    var testPattern: Binding<Bool> {
        Binding(get: { self.app.testPattern }, set: { self.app.testPattern = $0 })
    }
    var bypassLenticular: Binding<Bool> {
        Binding(get: { self.app.bypassLenticular }, set: { self.app.bypassLenticular = $0 })
    }
    var tileGrid: Binding<Bool> {
        Binding(get: { self.app.showTileGrid }, set: { self.app.showTileGrid = $0 })
    }
    var clockPaused: Binding<Bool> {
        Binding(get: { self.app.paused }, set: { self.app.setPaused($0) })
    }
    var renderScale: Binding<Float> {
        Binding(get: { self.app.renderer.renderScale }, set: { self.app.renderer.renderScale = $0 })
    }
}

// ---- 本地训练子进程 -----------------------------------------------------------

/// `uv run train <task> --env.scene.num-envs N --agent.max_iterations M`
///（microduck_rl 约定：先 64 envs × 5 iters smoke，再上长 run）。
/// 日志增量推入 ring buffer；解析 rsl_rl 的 "Mean reward:" 行成曲线数据。
@MainActor final class TrainingRunner: ObservableObject {
    enum State { case idle, running, exited(Int32) }
    @Published private(set) var state: State = .idle
    @Published private(set) var logTail = ""
    @Published private(set) var rewards: [Double] = []
    /// 自动续跑已完成的块数（每块 trainIters 轮）。
    @Published private(set) var chunks = 0

    private var proc: Process?
    private var buffer = ""
    private let maxChars = 20000
    private static let rewardRe = try! NSRegularExpression(
        pattern: #"Mean reward:\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"#)
    private var autoLoop = false
    private var lastConfig: (task: String, envs: Int, iters: Int, extraArgs: String,
                             workDir: String, cpuOnly: Bool, wandbOffline: Bool,
                             lkgStream: Bool)?

    var isRunning: Bool { if case .running = state { return true } else { return false } }

    /// workDir = microduck_rl 仓库根（含 pyproject/uv.lock）。
    /// autoLoop：每块结束后 --agent.resume True（默认取最新 run 的最新 checkpoint）
    /// 续跑下一块，直到 stop() 或非零退出。
    func start(task: String, envs: Int, iters: Int, extraArgs: String, workDir: String,
               cpuOnly: Bool, wandbOffline: Bool, lkgStream: Bool, autoLoop: Bool) {
        guard !isRunning else { return }
        self.autoLoop = autoLoop
        lastConfig = (task, envs, iters, extraArgs, workDir, cpuOnly, wandbOffline, lkgStream)
        chunks = 0
        rewards = []
        logTail = ""
        buffer = ""
        launch(task: task, envs: envs, iters: iters, extraArgs: extraArgs,
               workDir: workDir, cpuOnly: cpuOnly, wandbOffline: wandbOffline,
               lkgStream: lkgStream, resume: false)
    }

    func stop() {
        autoLoop = false
        guard let p = proc, p.isRunning else { return }
        p.terminate()   // SIGTERM；mjlab/rsl_rl 负责清理子进程
    }

    private func terminated(_ code: Int32) {
        proc = nil
        if code == 0 { chunks += 1 }
        if autoLoop, code == 0, let cfg = lastConfig {
            buffer += "\n===== chunk \(chunks) done，2s 后 resume 续跑 =====\n"
            logTail = String(buffer.suffix(4000))
            let cfgCopy = cfg
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self, self.autoLoop else { return }
                self.launch(task: cfgCopy.task, envs: cfgCopy.envs, iters: cfgCopy.iters,
                            extraArgs: cfgCopy.extraArgs, workDir: cfgCopy.workDir,
                            cpuOnly: cfgCopy.cpuOnly, wandbOffline: cfgCopy.wandbOffline,
                            lkgStream: cfgCopy.lkgStream, resume: true)
            }
        } else {
            autoLoop = false
            state = .exited(code)
        }
    }

    private func launch(task: String, envs: Int, iters: Int, extraArgs: String,
                        workDir: String, cpuOnly: Bool, wandbOffline: Bool,
                        lkgStream: Bool, resume: Bool) {
        var args = ["run", "train", task,
                    "--env.scene.num-envs", String(envs),
                    "--agent.max_iterations", String(iters)]
        if cpuOnly { args += ["--gpu-ids", "None"] }
        if resume { args += ["--agent.resume", "True"] }   // 默认取最新 run 最新 checkpoint
        args += extraArgs.split(separator: " ").map(String.init)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["uv"] + args
        p.currentDirectoryURL = URL(fileURLWithPath: workDir)
        // 本机仓库在 iCloud Documents 同步域里，fileproviderd 会给 venv 里的
        // .pth 打 UF_HIDDEN，而 uv 版 CPython 的 site.py 跳过 hidden .pth →
        // editable 的 src/ 链接失效（ModuleNotFoundError: mjlab_microduck）。
        // 用 PYTHONPATH 显式补 src，与 .pth 状态无关。
        var env = ProcessInfo.processInfo.environment
        let srcPath = workDir + "/src"
        env["PYTHONPATH"] = env["PYTHONPATH"].map { srcPath + ":" + $0 } ?? srcPath
        // venv 放 iCloud 同步域外（~/Documents 下的 .venv 会被 fileproviderd
        // 打 UF_HIDDEN，uv 版 CPython 跳过 hidden .pth → 训练随机起不来）。
        env["UV_PROJECT_ENVIRONMENT"] = "/Users/jup33q/kimi-work/venvs/microduck_rl"
        if wandbOffline { env["WANDB_MODE"] = "offline" }
        if lkgStream { env["MICRODUCK_LKG_STREAM"] = "44123" }
        env["MJLAB_WARP_QUIET"] = "1"   // 日志尾留给迭代/指标行
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        p.terminationHandler = { [weak self] pr in
            Task { @MainActor in self?.terminated(pr.terminationStatus) }
        }
        pipe.fileHandleForReading.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.ingest(s) }
        }
        do {
            try p.run()
            proc = p
            state = .running
        } catch {
            buffer += "launch failed: \(error.localizedDescription)\n"
            logTail = String(buffer.suffix(4000))
            autoLoop = false
            state = .exited(-1)
        }
    }

    private func ingest(_ s: String) {
        buffer += s
        if buffer.count > maxChars { buffer = String(buffer.suffix(maxChars)) }
        logTail = String(buffer.suffix(4000))
        let ns = s as NSString
        for m in Self.rewardRe.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            if let v = Double(ns.substring(with: m.range(at: 1))) { rewards.append(v) }
        }
    }
}

// ---- 面板视图 -----------------------------------------------------------------

@MainActor
struct ControlPanelView: View {
    @ObservedObject var control: ViewerControl
    @StateObject private var training = TrainingRunner()

    var body: some View {
        Form {
            Section("遥测") {
                LabeledContent("FPS", value: String(format: "%.1f", control.fps))
                LabeledContent("|qpos|", value: String(format: "%.4f", control.qpos))
                LabeledContent("sim t", value: String(format: "%.2f s", control.simT))
                LabeledContent("draws / tris", value: "\(control.draws) / \(control.tris)")
                Text(control.distInfo).font(.caption).foregroundStyle(.secondary)
            }
            Section("相机") {
                HStack {
                    Text("fov")
                    Slider(value: $control.fovDeg, in: 15...60)
                    Text("\(Int(control.fovDeg.rounded()))°").monospacedDigit().frame(width: 34)
                }
                HStack {
                    Text("方位角")
                    Slider(value: $control.azDeg, in: 0...360)
                    Text("\(Int(control.azDeg.rounded()))°").monospacedDigit().frame(width: 34)
                }
                HStack {
                    Text("距离倍率")
                    Slider(value: $control.margin, in: 0.6...2.5)
                    Text(String(format: "%.2f", control.margin)).monospacedDigit().frame(width: 34)
                }
                HStack {
                    Text("目标偏移")
                    Slider(value: $control.offsetX, in: -1...1)
                    Slider(value: $control.offsetY, in: -1...1)
                }
                Toggle("平滑跟随", isOn: $control.followSmooth)
                Toggle("朝向跟随（转身相机摇到身后）", isOn: $control.followHeading)
            }
            Section("显示") {
                Toggle("校准测试图 (c)", isOn: control.testPattern)
                Toggle("裸 quilt 上屏 (b)", isOn: control.bypassLenticular)
                Toggle("预览 tile 网格 (g)", isOn: control.tileGrid)
                Toggle("暂停时钟 (p)", isOn: control.clockPaused)
                Picker("quilt 渲染分辨率", selection: control.renderScale) {
                    Text("全分辨率").tag(Float(1))
                    Text("半分辨率").tag(Float(0.5))
                }
                HStack {
                    Button("存 quilt PNG") { control.saveQuilt() }
                    Button("存交织 PNG") { control.saveLenticular() }
                }
            }
            Section("物理") {
                Toggle("暂停物理", isOn: $control.physicsPaused)
                Stepper("每帧 mj_step ×\(control.stepsPerFrame)",
                        value: $control.stepsPerFrame, in: 0...8)
                Button("重置到 keyframe 0") { control.resetRequest &+= 1 }
            }
            Section("策略驾驶（P3 · BAM M6 执行器）") {
                Picker("模式", selection: $control.driveMode) {
                    ForEach(ViewerControl.DriveMode.allCases, id: \.self) {
                        Text($0.rawValue).tag($0)
                    }
                }
                .pickerStyle(.segmented)
                if control.driveMode == .policy {
                    Toggle("BAM 执行器（训练同款 M6 电压模型）", isOn: $control.actuatorBAM)
                    Text("关 = XML PD（--no-bam 语义）。切换会重置鸭子姿态")
                        .font(.caption2).foregroundStyle(.secondary)
                    Picker("策略", selection: $control.policyPath) {
                        ForEach(ViewerControl.availablePolicies, id: \.self) {
                            Text($0.split(separator: "/").last.map(String.init) ?? $0).tag($0)
                        }
                    }
                    Stepper("鸭子数 ×\(control.duckCount)", value: $control.duckCount, in: 1...8)
                    Text("多鸭子顶点吞吐 ×N，>3 只建议半分辨率（显示区可切）")
                        .font(.caption2).foregroundStyle(.secondary)
                    HStack {
                        Text("前进")
                        Slider(value: $control.cmdVX, in: -0.3...0.3)
                        Text(String(format: "%+.2f", control.cmdVX)).monospacedDigit().frame(width: 44)
                    }
                    HStack {
                        Text("侧移")
                        Slider(value: $control.cmdVY, in: -0.2...0.2)
                        Text(String(format: "%+.2f", control.cmdVY)).monospacedDigit().frame(width: 44)
                    }
                    HStack {
                        Text("转向")
                        Slider(value: $control.cmdYaw, in: -1.5...1.5)
                        Text(String(format: "%+.2f", control.cmdYaw)).monospacedDigit().frame(width: 44)
                    }
                    HStack {
                        Text("头俯仰")
                        Slider(value: $control.headPitchDeg, in: -35...35)
                        Text(String(format: "%+.0f°", control.headPitchDeg)).monospacedDigit().frame(width: 44)
                    }
                    HStack {
                        Text("头偏航")
                        Slider(value: $control.headYawDeg, in: -60...60)
                        Text(String(format: "%+.0f°", control.headYawDeg)).monospacedDigit().frame(width: 44)
                    }
                    Text("俯仰 + = 低头 · 偏航 + = 左转（quackd look 语义，限位 ±35°/±60°）")
                        .font(.caption2).foregroundStyle(.secondary)
                    HStack {
                        Button("指令清零 (X)") {
                            control.cmdVX = 0; control.cmdVY = 0; control.cmdYaw = 0
                            control.headPitchDeg = 0; control.headYawDeg = 0
                        }
                        Button("推倒（空格）") { control.pushRequest &+= 1 }
                    }
                    Text("W/S 前进后退 · A/D 转向 · Q/E 侧移 · X 清零 · 空格推倒")
                        .font(.caption2).foregroundStyle(.secondary)
                    if control.vlmArenaOK && LayaSidecar.available {
                        Divider()
                        Toggle("VLM 驾驶（laya 找球→踢球）", isOn: $control.vlmDrive)
                        if control.vlmDrive {
                            HStack(alignment: .top) {
                                if let f = control.vlmFrame {
                                    Image(nsImage: f)
                                        .resizable()
                                        .interpolation(.none)
                                        .frame(width: 128, height: 128)
                                        .border(.quaternary)
                                }
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(control.vlmStatus)
                                        .font(.system(.caption, design: .monospaced))
                                    Text("左图 = VLM 所见（头相机 256²）。决策 ~1-2Hz，"
                                         + "KICK 为 quackd 脚本推球（0.3m/±35° 锥）")
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            if AppRes.rlRepo != nil {
                Section("本地训练（microduck_rl · uv run train）") {
                TextField("task id", text: $control.trainTask)
                    .font(.system(.body, design: .monospaced))
                HStack {
                    Stepper("envs \(control.trainEnvs)", value: $control.trainEnvs, in: 1...4096)
                    Stepper("iters \(control.trainIters)", value: $control.trainIters, in: 1...10000)
                }
                TextField("额外参数（可空）", text: $control.trainExtraArgs)
                    .font(.system(.caption, design: .monospaced))
                HStack {
                    Picker("设备", selection: $control.trainDeviceCPU) {
                        Text("CPU（本机）").tag(true)
                        Text("CUDA").tag(false)
                    }
                    .pickerStyle(.segmented)
                    Toggle("wandb 离线", isOn: $control.trainWandbOffline)
                }
                Toggle("LKG 直播（viewer --stream 44123）", isOn: $control.trainLkgStream)
                HStack {
                    Toggle("自动续跑", isOn: $control.trainAutoLoop)
                    if training.chunks > 0 {
                        Text("已完成 \(training.chunks) 块")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Circle()
                        .fill(training.isRunning ? .green : .gray)
                        .frame(width: 8, height: 8)
                    Button(training.isRunning ? "停止" : "开始训练") {
                        if training.isRunning {
                            training.stop()
                        } else {
                            training.start(task: control.trainTask,
                                           envs: control.trainEnvs,
                                           iters: control.trainIters,
                                           extraArgs: control.trainExtraArgs,
                                           workDir: control.trainingWorkDir,
                                           cpuOnly: control.trainDeviceCPU,
                                           wandbOffline: control.trainWandbOffline,
                                           lkgStream: control.trainLkgStream,
                                           autoLoop: control.trainAutoLoop)
                        }
                    }
                    Spacer()
                    if let last = training.rewards.last {
                        Text(String(format: "reward %.3f (×%d)", last, training.rewards.count))
                            .font(.caption).monospacedDigit()
                    }
                }
                if training.rewards.count > 1 {
                    RewardSparkline(values: training.rewards)
                        .frame(height: 40)
                }
                ScrollView {
                    Text(training.logTail)
                        .font(.system(size: 9, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 150)
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 360)
    }
}

/// rsl_rl Mean reward 折线（归一化到 min...max）。
struct RewardSparkline: View {
    let values: [Double]
    var body: some View {
        Canvas { ctx, size in
            guard let lo = values.min(), let hi = values.max(), hi > lo else { return }
            var path = Path()
            for (i, v) in values.enumerated() {
                let x = size.width * Double(i) / Double(values.count - 1)
                let y = size.height * (1 - (v - lo) / (hi - lo))
                if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                else { path.addLine(to: CGPoint(x: x, y: y)) }
            }
            ctx.stroke(path, with: .color(.accentColor), lineWidth: 1.5)
        }
    }
}

extension ViewerControl {
    func saveQuilt() {
        // P5：落桌面（.app 双击时 cwd 是 "/"，写到那里会权限失败）
        let path = NSHomeDirectory() + "/Desktop/quilt\(app.renderer.spec.namingSuffix).png"
        app.saveQuilt(to: path)
    }
    func saveLenticular() {
        let path = NSHomeDirectory() + "/Desktop/lenticular.png"
        app.saveLenticular(to: path)
    }

    /// 主屏控制面板窗口（普通 titled 窗，仅在主屏；不影响 LKG 设备窗焦点行为）。
    func makePanelWindow() -> NSWindow {
        let w = NSWindow(contentRect: NSRect(x: 560, y: 120, width: 380, height: 780),
                         styleMask: [.titled, .closable, .miniaturizable],
                         backing: .buffered, defer: false)
        w.title = "MuJoCo-LKG 控制面板"
        w.contentView = NSHostingView(rootView: ControlPanelView(control: self))
        w.isReleasedWhenClosed = false
        return w
    }
}
