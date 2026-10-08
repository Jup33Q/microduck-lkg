import AppKit
import Foundation

// P3-VLM：laya-vision-microduck-kick 高层驾驶 sidecar。
// 子进程跑 laya_sidecar.py（torch/MPS，~50ms/决策），stdin 收 256×256 RGB 帧，
// stdout 回 "ACT <choice> <pF> <pL> <pR> <pK> <ms>"。协议见 laya_sidecar.py。
// 单 in-flight：上一帧没答完就不发新帧（决策天然 1-2Hz 节流）。

@MainActor
final class LayaSidecar {
    struct Decision {
        var choice: String   // FORWARD / LEFT / RIGHT / KICK
        var probs: (f: Float, l: Float, r: Float, k: Float)
        var ms: Int
    }

    private var proc: Process?
    private var stdinFH: FileHandle?
    private(set) var ready = false
    private(set) var inFlight = false
    private var inFlightSince = 0.0
    private var lineBuf = Data()
    var isRunning: Bool { proc?.isRunning ?? false }
    var onDecision: ((Decision) -> Void)?
    var onLog: ((String) -> Void)?

    static let pythonPath = "/Users/jup33q/kimi-work/venvs/laya/bin/python"
    static let scriptPath = "/Users/jup33q/kimi-work/laya-work/laya_sidecar.py"

    /// sidecar 环境在位（P5：.app 在别的机器上没这环境，面板隐藏 VLM 开关）
    static var available: Bool {
        FileManager.default.fileExists(atPath: pythonPath)
            && FileManager.default.fileExists(atPath: scriptPath)
    }

    func start() {
        guard proc == nil else { return }
        guard FileManager.default.fileExists(atPath: Self.pythonPath),
              FileManager.default.fileExists(atPath: Self.scriptPath) else {
            onLog?("vlm: sidecar 不存在（\(Self.pythonPath)）")
            return
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: Self.pythonPath)
        p.arguments = [Self.scriptPath]
        let inPipe = Pipe()
        let outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        // stderr 落文件（sidecar 挂了好排查；/dev/null 会把崩溃 traceback 吞了）
        let errPath = "/tmp/laya_sidecar.err"
        FileManager.default.createFile(atPath: errPath, contents: nil)
        if let eh = FileHandle(forWritingAtPath: errPath) { p.standardError = eh }
        else { p.standardError = FileHandle.nullDevice }
        p.terminationHandler = { [weak self] pr in
            // 直接 print：weak self 失效时也得留下记录（静默死进程排查）
            print("vlm: sidecar 退出 code=\(pr.terminationStatus) reason=\(pr.terminationReason.rawValue)")
            Task { @MainActor in
                self?.proc = nil
                self?.ready = false
                self?.inFlight = false
            }
        }
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor in self?.ingest(data) }
        }
        do {
            try p.run()
            proc = p
            stdinFH = inPipe.fileHandleForWriting
            onLog?("vlm: sidecar 启动中（模型加载 ~10s）…")
        } catch {
            onLog?("vlm: sidecar 启动失败 \(error.localizedDescription)")
        }
    }

    func stop() {
        proc?.terminate()
        proc = nil
        ready = false
        inFlight = false
    }

    /// 送一帧 256×256 RGB（196608 字节）。未 ready 或有 in-flight 时丢弃（返回 false）。
    @discardableResult
    func send(rgb: [UInt8]) -> Bool {
        guard ready, !inFlight, let fh = stdinFH, rgb.count == 256 * 256 * 3 else { return false }
        inFlight = true
        inFlightSince = CACurrentMediaTime()
        do {
            try fh.write(contentsOf: Data("FRAME\n".utf8))
            try fh.write(contentsOf: rgb)
        } catch {
            // 子进程已死（管道断裂）：丢掉 in-flight，等 terminationHandler 清理
            inFlight = false
            onLog?("vlm: 写帧失败（\(error.localizedDescription)）")
            return false
        }
        return true
    }

    /// in-flight 看门狗（渲染循环每帧调用）：sidecar 答不出（>3s）时解锁，
    /// 避免一次失败永久卡住查询链。
    func watchdog() {
        if inFlight && CACurrentMediaTime() - inFlightSince > 3 {
            inFlight = false
            onLog?("vlm: 决策超时（>3s），解锁重试")
        }
    }

    private func ingest(_ data: Data) {
        lineBuf.append(data)
        while let nl = lineBuf.firstIndex(of: 0x0a) {
            let line = lineBuf.subdata(in: 0..<nl)
            lineBuf.removeSubrange(...nl)
            guard let s = String(data: line, encoding: .utf8) else { continue }
            if s == "READY" {
                ready = true
                onLog?("vlm: sidecar READY")
            } else if s.hasPrefix("ACT ") {
                let parts = s.split(separator: " ")
                if parts.count >= 7, let pf = Float(parts[2]), let pl = Float(parts[3]),
                   let pr = Float(parts[4]), let pk = Float(parts[5]), let ms = Int(parts[6]) {
                    inFlight = false
                    onDecision?(Decision(choice: String(parts[1]),
                                         probs: (pf, pl, pr, pk), ms: ms))
                } else {
                    inFlight = false
                }
            } else if s.hasPrefix("ERR ") {
                inFlight = false
                onLog?("vlm: \(s)")
            }
        }
    }
}
