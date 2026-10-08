import Foundation
import MuJoCo
import COnnxRuntime

// P3: RL 策略本地推理（onnxruntime，与 infer_policy.py 同运行时；normalizer 已烘焙
// 进 ONNX，喂原始 obs）。执行器默认 BAM M6（BamController.swift，训练同款电压模型，
// = infer_policy 默认路径），useBam=false 时退回 XML position PD（--no-bam 语义）。
// 契约逐项对齐 microduck_rl/scripts/infer_policy.py（obs 61D = 3+3+14+14+14+13，
// 动作 = DEFAULT_POSE + action·1.0，无滤波无裁剪，50Hz 策略 / 200Hz 物理 decimation 4）。

final class PolicyRunner {
    /// STAND2 姿态（= infer_policy DEFAULT_POSE = HOME_FRAME = scene.xml STAND keyframe）
    static let defaultPose: [Float] = [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,   // 左腿：hip_yaw/roll/pitch, knee, ankle
        0.3491, 0.3491, 0.0, 0.0,                  // 颈/头：neck_pitch, head_pitch/yaw/roll
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,     // 右腿
    ]

    /// 13D 命令块：[twist(3)=vx,vy,vyaw] + [head_pose(4)] + [body_pose(6)]
    struct Command {
        var vx: Float = 0, vy: Float = 0, vyaw: Float = 0
        var head = SIMD4<Float>(0, 0, 0, 0)
        var body = (SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0))
    }

    private let m: UnsafeMutablePointer<mjModel>
    private let d: UnsafeMutablePointer<mjData>

    // ORT
    private var ortEnv: OpaquePointer?      // OrtEnv
    private var ortSession: OpaquePointer?  // OrtSession
    private var ortMem: OpaquePointer?      // OrtMemoryInfo
    private var inTensor: OpaquePointer?    // OrtValue
    private var obsBuf = [Float](repeating: 0, count: 61)

    // obs 索引
    private var jointQposAdr = [Int]()
    private var jointDofAdr = [Int]()
    private var imuAngVelAdr = -1
    private var trunkBodyId = -1
    private var trunkQposAdr = -1
    private var trunkDofAdr = -1

    /// BAM M6 执行器（nil = XML PD / --no-bam 语义）
    let bam: BamController?

    var command = Command()
    private var lastAction = [Float](repeating: 0, count: 14)
    /// 策略节拍累积器（50Hz）
    private var acc: Double = 0
    private var diagPrinted = false
    // achieved/cmd 遥测（infer_policy [vel 1s avg] 同款：body 系 1s 窗口均值）
    private var tickCount = 0
    private var velWin: [(Float, Float, Float)] = []

    init?(m: UnsafeMutablePointer<mjModel>, d: UnsafeMutablePointer<mjData>,
          onnxPath: String, useBam: Bool = false) {
        self.m = m
        self.d = d
        let mm = m.pointee

        // 关节索引：actuator_trnid 反查（不硬编码 qpos 下标）
        let nu = Int(mm.nu)
        guard nu == 14 else { print("policy: 期望 14 执行器，实际 nu=\(nu)"); return nil }
        for i in 0..<nu {
            let jnt = Int(mm.actuator_trnid[i * 2])
            jointQposAdr.append(Int(mm.jnt_qposadr[jnt]))
            jointDofAdr.append(Int(mm.jnt_dofadr[jnt]))
        }
        var sid = mj_name2id(m, Int32(mjOBJ_SENSOR.rawValue), "imu_ang_vel")
        if sid >= 0 { imuAngVelAdr = Int(mm.sensor_adr[Int(sid)]) }
        sid = mj_name2id(m, Int32(mjOBJ_BODY.rawValue), "trunk_base")
        if sid >= 0 { trunkBodyId = Int(sid) }
        let fjid = mj_name2id(m, Int32(mjOBJ_JOINT.rawValue), "trunk_base_freejoint")
        if fjid >= 0 {
            trunkQposAdr = Int(mm.jnt_qposadr[Int(fjid)])
            trunkDofAdr = Int(mm.jnt_dofadr[Int(fjid)])
        }
        guard imuAngVelAdr >= 0, trunkBodyId >= 0, trunkDofAdr >= 0 else {
            print("policy: 缺 imu_ang_vel 传感器或 trunk_base body/freejoint"); return nil
        }
        if useBam {
            guard let b = BamController(m: m, d: d) else { return nil }
            bam = b
        } else {
            bam = nil
        }

        // ORT 初始化
        guard let apiBase = OrtGetApiBase() else { return nil }
        let api = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION))
        guard let api else { print("policy: ORT GetApi failed"); return nil }
        var status = api.pointee.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mjlkg", &ortEnv)
        guard status == nil else { print("policy: CreateEnv failed"); return nil }
        var opts: OpaquePointer?
        status = api.pointee.CreateSessionOptions(&opts)
        guard status == nil, let opts else { return nil }
        status = api.pointee.CreateSession(ortEnv, onnxPath, opts, &ortSession)
        api.pointee.ReleaseSessionOptions(opts)
        guard status == nil, ortSession != nil else {
            let msg = status.flatMap { api.pointee.GetErrorMessage($0) }.map { String(cString: $0) } ?? "?"
            print("policy: CreateSession failed: \(msg)")
            return nil
        }
        status = api.pointee.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ortMem)
        guard status == nil else { return nil }
        // 输入 tensor 复用：1×61 f32
        var shape: [Int64] = [1, 61]
        status = obsBuf.withUnsafeMutableBufferPointer { buf in
            shape.withUnsafeMutableBufferPointer { sh in
                api.pointee.CreateTensorWithDataAsOrtValue(
                    ortMem, buf.baseAddress, 61 * MemoryLayout<Float>.stride,
                    sh.baseAddress, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inTensor)
            }
        }
        guard status == nil, inTensor != nil else { print("policy: CreateTensor failed"); return nil }
    }

    /// 初始状态：优先 STAND keyframe（模型无关——rollers 身高 0.1385 ≠ walk 0.125，
    /// 等价于 infer_policy 手写 STAND2 的效果）；无 keyframe 时退回 DEFAULT_POSE 手写。
    /// 无 reset 逻辑（单 episode 无限跑，摔了躺平）。
    func resetPose() {
        if m.pointee.nkey > 0 {
            var keyId: Int32 = 0
            let sid = mj_name2id(m, Int32(mjOBJ_KEY.rawValue), "STAND")
            if sid >= 0 { keyId = sid }
            mj_resetDataKeyframe(m, d, keyId)
        } else {
            for i in 0..<Int(m.pointee.nq) { d.pointee.qpos[i] = 0 }
            d.pointee.qpos[2] = 0.125
            d.pointee.qpos[3] = 1   // quat w
            for i in 0..<14 { d.pointee.qpos[jointQposAdr[i]] = Double(Self.defaultPose[i]) }
        }
        for i in 0..<Int(m.pointee.nv) { d.pointee.qvel[i] = 0 }
        if let bam {
            // BAM：ctrl 是电机扭矩，由 update() 每子步覆写；参考实现 reset 时 ctrl=0
            //（mj_resetData 语义），q_target = 当前关节位
            for i in 0..<Int(m.pointee.nu) { d.pointee.ctrl[i] = 0 }
            bam.reset()
        } else {
            for i in 0..<14 { d.pointee.ctrl[i] = d.pointee.qpos[jointQposAdr[i]] }
        }
        lastAction = [Float](repeating: 0, count: 14)
        acc = 0
        tickCount = 0
        velWin.removeAll()
        mj_forward(m, d)
    }

    /// projected_gravity = quat_rotate_inverse(xquat[trunk], (0,0,-1))，MuJoCo wxyz
    private func projectedGravity() -> (Float, Float, Float) {
        guard let xquat = d.pointee.xquat else { return (0, 0, -1) }
        let b = trunkBodyId * 4
        let w = Float(xquat[b]), x = Float(xquat[b + 1]),
            y = Float(xquat[b + 2]), z = Float(xquat[b + 3])
        // t = 2·cross(q.xyz, v); out = v − w·t + cross(q.xyz, t); v = (0,0,-1)
        let tx = 2 * (y * -1.0 - z * 0), ty = 2 * (z * 0 - x * -1.0), tz: Float = 0
        let rx = 0 - w * tx + (y * tz - z * ty)
        let ry = 0 - w * ty + (z * tx - x * tz)
        let rz = -1 - w * tz + (x * ty - y * tx)
        return (rx, ry, rz)
    }

    private func writeObs() {
        guard let sd = d.pointee.sensordata else { return }
        let g = projectedGravity()
        obsBuf[0] = Float(sd[imuAngVelAdr]); obsBuf[1] = Float(sd[imuAngVelAdr + 1]); obsBuf[2] = Float(sd[imuAngVelAdr + 2])
        obsBuf[3] = g.0; obsBuf[4] = g.1; obsBuf[5] = g.2
        for i in 0..<14 {
            obsBuf[6 + i] = Float(d.pointee.qpos[jointQposAdr[i]]) - Self.defaultPose[i]
            obsBuf[20 + i] = Float(d.pointee.qvel[jointDofAdr[i]])
            obsBuf[34 + i] = lastAction[i]
        }
        obsBuf[48] = command.vx; obsBuf[49] = command.vy; obsBuf[50] = command.vyaw
        obsBuf[51] = command.head.x; obsBuf[52] = command.head.y
        obsBuf[53] = command.head.z; obsBuf[54] = command.head.w
        obsBuf[55] = command.body.0.x; obsBuf[56] = command.body.0.y; obsBuf[57] = command.body.0.z
        obsBuf[58] = command.body.1.x; obsBuf[59] = command.body.1.y; obsBuf[60] = command.body.1.z
    }

    /// quat_rotate_inverse（wxyz）：v − w·t + cross(q.xyz, t)，t = 2·cross(q.xyz, v)
    private static func quatRotateInverse(_ w: Float, _ x: Float, _ y: Float, _ z: Float,
                                          _ vx: Float, _ vy: Float, _ vz: Float) -> (Float, Float, Float) {
        let tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx)
        return (vx - w * tx + (y * tz - z * ty),
                vy - w * ty + (z * tx - x * tz),
                vz - w * tz + (x * ty - y * tx))
    }

    /// infer_policy [vel 1s avg] 同款遥测：body 系 fwd/lat/yaw 的 1s（50 拍）窗口均值 vs 指令
    private func trackVelocity() {
        let qa = trunkQposAdr, da = trunkDofAdr
        let qw = Float(d.pointee.qpos[qa + 3]), qx = Float(d.pointee.qpos[qa + 4]),
            qy = Float(d.pointee.qpos[qa + 5]), qz = Float(d.pointee.qpos[qa + 6])
        let vb = Self.quatRotateInverse(qw, qx, qy, qz,
                                        Float(d.pointee.qvel[da]), Float(d.pointee.qvel[da + 1]),
                                        Float(d.pointee.qvel[da + 2]))
        velWin.append((vb.0, vb.1, Float(d.pointee.qvel[da + 5])))
        tickCount += 1
        guard tickCount % 50 == 0, !velWin.isEmpty else { return }
        let n = Float(velWin.count)
        let avg = velWin.reduce((0, 0, 0) as (Float, Float, Float)) { ($0.0 + $1.0, $0.1 + $1.1, $0.2 + $1.2) }
        print(String(format: "[vel 1s avg] achieved/cmd  fwd=%+.2f/%+.2f  lat=%+.2f/%+.2f m/s  yaw=%+.2f/%+.2f rad/s   trunk_z=%.1f mm  (%@)",
                     avg.0 / n, command.vx, avg.1 / n, command.vy, avg.2 / n, command.vyaw,
                     Float(d.pointee.qpos[qa + 2]) * 1000, bam != nil ? "bam" : "xmlpd"))
        velWin.removeAll()
    }

    /// 一个 50Hz 策略拍：obs → ONNX → 动作写入（BAM: q_target；XML PD: ctrl 目标位）。
    /// 物理子步由调用方执行 stepPhysics() ×4（多鸭子交错 BAM update/mj_step 用）。
    /// 返回是否执行了一拍（按 wall-dt 累积凑 50Hz）。
    @discardableResult
    func tick(wallDt: Double) -> Bool {
        acc += wallDt
        guard acc >= 0.02 else { return false }
        acc = acc >= 0.04 ? 0 : acc - 0.02   // 落后多时只补一拍，不追尾
        trackVelocity()
        writeObs()
        guard let apiBase = OrtGetApiBase(),
              let api = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION)),
              let ortSession, let inTensor else { return false }
        var out: OpaquePointer?
        var inNameBuf: [CChar] = Array("obs".utf8CString)
        var outNameBuf: [CChar] = Array("actions".utf8CString)
        var status: OpaquePointer? = nil
        inNameBuf.withUnsafeBufferPointer { nb in
            outNameBuf.withUnsafeBufferPointer { ob in
                var inNames: [UnsafePointer<CChar>?] = [nb.baseAddress]
                var outNames: [UnsafePointer<CChar>?] = [ob.baseAddress]
                var inputs: [OpaquePointer?] = [inTensor]
                inNames.withUnsafeBufferPointer { inp in
                    outNames.withUnsafeBufferPointer { onp in
                        inputs.withUnsafeBufferPointer { ip in
                            status = api.pointee.Run(ortSession, nil,
                                                     inp.baseAddress, ip.baseAddress, 1,
                                                     onp.baseAddress, 1, &out)
                        }
                    }
                }
            }
        }
        guard status == nil, let out else {
            if !diagPrinted {
                diagPrinted = true
                let msg = status.flatMap { api.pointee.GetErrorMessage($0) }.map { String(cString: $0) } ?? "nil-out"
                print("policy: Run failed: \(msg)")
            }
            return false
        }
        var outPtr: UnsafeMutableRawPointer?
        status = api.pointee.GetTensorMutableData(out, &outPtr)
        guard status == nil, let outPtr else { api.pointee.ReleaseValue(out); return false }
        if !diagPrinted {
            diagPrinted = true
            let p = outPtr.assumingMemoryBound(to: Float.self)
            print(String(format: "policy: first tick OK, action[0..3] = %.3f %.3f %.3f | obs[0..5] = %.3f %.3f %.3f %.3f %.3f %.3f",
                         p[0], p[1], p[2], obsBuf[0], obsBuf[1], obsBuf[2], obsBuf[3], obsBuf[4], obsBuf[5]))
        }
        let act = outPtr.assumingMemoryBound(to: Float.self)
        for i in 0..<14 {
            let a = act[i]
            lastAction[i] = a
            if let bam {
                bam.qTarget[i] = Double(Self.defaultPose[i] + a)   // action_scale = 1.0
            } else {
                d.pointee.ctrl[i] = Double(Self.defaultPose[i] + a)
            }
        }
        api.pointee.ReleaseValue(out)
        return true
    }

    /// 一个 200Hz 物理子步：BAM 时先 update（duty→V→τ + 摩擦重写）再 mj_step。
    /// 多鸭子须逐只 update→step 交错（摩擦参数写在共享 mjModel 上，last-writer-wins）。
    func stepPhysics() {
        bam?.update()
        mj_step(m, d)
    }
}
