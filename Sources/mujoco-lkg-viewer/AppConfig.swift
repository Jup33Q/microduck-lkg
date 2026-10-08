import Foundation

// P5: 资源路径解硬编码 + 会话配置持久化（桌面 .app 封装）。
// 解析顺序：① app bundle Resources（打包产物）② 开发仓库路径 ③ 兜底。
// 训练/VLM sidecar 依赖外部环境（microduck_rl repo / ~/kimi-work venv），
// 只存在性探测、缺失则面板隐藏对应区，不阻塞主流程。

enum AppRes {
    static let devRoot = "/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360"
    static let devRobotDir = devRoot + "/microduck_rl/src/mjlab_microduck/robot/microduck"
    static let devScenesDir = devRoot + "/mujoco-lkg/scenes"
    static let devPoliciesDir = devRoot + "/mujoco-lkg/models/pollen-policies"
    static let devRlRepo = devRoot + "/microduck_rl"

    private static var fm: FileManager { FileManager.default }

    private static func firstExisting(_ candidates: [String?]) -> String? {
        for c in candidates { if let c, fm.fileExists(atPath: c) { return c } }
        return nil
    }

    /// 机器人目录（robot_groundcontact.xml + assets/）
    static var robotDir: String {
        firstExisting([Bundle.main.resourcePath.map { $0 + "/robot/robot_groundcontact.xml" },
                       devRobotDir + "/robot_groundcontact.xml"])
            .map { $0.replacingOccurrences(of: "/robot_groundcontact.xml", with: "") }
            ?? devRobotDir
    }

    /// 默认场景：竞技场（含球/墙/head_camera，VLM 全流程可演示）
    static var defaultScene: String {
        firstExisting([Bundle.main.resourcePath.map { $0 + "/scenes/arena_ball.xml" },
                       devScenesDir + "/arena_ball.xml"])
            ?? devRobotDir + "/scene.xml"
    }

    /// pollen-policies 目录（10 个官方 ONNX）
    static var policiesDir: String? {
        firstExisting([Bundle.main.resourcePath.map { $0 + "/policies" }, devPoliciesDir])
    }

    /// microduck_rl 仓库（本地训练面板；不存在则隐藏该区）
    static var rlRepo: String? {
        fm.fileExists(atPath: devRlRepo + "/pyproject.toml") ? devRlRepo : nil
    }

    /// 可选策略清单：policiesDir 全量 + 开发仓库根的 m1 系列（存在才列）
    static var availablePolicies: [String] {
        var out: [String] = []
        if let dir = policiesDir,
           let files = try? fm.contentsOfDirectory(atPath: dir) {
            for f in files.sorted() where f.hasSuffix(".onnx") { out.append(dir + "/" + f) }
        }
        for f in ["microduck_m1_800.onnx", "microduck_m1_5_2000.onnx", "microduck_smoke.onnx"] {
            let p = devRoot + "/microduck_rl/" + f
            if fm.fileExists(atPath: p) { out.append(p) }
        }
        return out
    }

    static var defaultPolicy: String {
        if let dir = policiesDir,
           fm.fileExists(atPath: dir + "/alpha_walking.onnx") { return dir + "/alpha_walking.onnx" }
        return availablePolicies.first ?? ""
    }
}

// ---- 会话配置（双击启动恢复上次；~/.mujoco-lkg/last-session.json） -------------

struct SessionConfig: Codable {
    var scenePath: String?
    var layout: String?
    var policyMode: Bool?
    var policyPath: String?
    var bam: Bool?
    var ducks: Int?
    var vlm: Bool?
    var fovDeg: Double?
    var azDeg: Double?
    var margin: Double?
    var followSmooth: Bool?
    var followHeading: Bool?
    var headPitchDeg: Double?
    var headYawDeg: Double?

    static var url: URL {
        let dir = NSHomeDirectory() + "/.mujoco-lkg"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return URL(fileURLWithPath: dir + "/last-session.json")
    }
    static func load() -> SessionConfig? {
        guard let d = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SessionConfig.self, from: d)
    }
    func save() {
        if let d = try? JSONEncoder().encode(self) { try? d.write(to: Self.url) }
    }
}
