import Foundation
import MuJoCo

// BAM M6（XL330 电压控制模型）执行器 —— bam.mujoco.MujocoController 的 Swift 移植。
// 参考实现：microduck_rl venv 的 bam 包（mujoco.py update() /
// actuator.py VoltageControlledActuator / model.py m6 compute_frictions），
// 常量来自 bam/params/xl330/m6.json 与 infer_policy.py 的 BAM_* 块（镜像训练侧
// microduck_constants._BAM_ACTUATOR_KWARGS，tests/test_infer_policy_bam.py 锁定）。
// 逐方程对齐 CPU 参考（infer_policy 默认 BAM 路径），包括其 efc_id↔关节 id 的
// 比较方式（mujoco 3.10 里 FRICTION_DOF 行的 efc_id 实际是 dof id，参考实现按
// 关节 id 匹配 → 错位归属是参考语义的一部分，原样保留，勿"修正"）。

/// XL330 M6 常量（bam/params/xl330/m6.json + infer_policy BAM_* 默认值）
enum BamM6 {
    static let kt = 0.36601349688984386        // 扭矩常数 [Nm/A]
    static let R = 2.8113923539223227          // 电机电阻 [Ω]
    static let armature = 0.0018077432831600838  // 转子附加惯量 [kg·m²]
    static let kpFW = 200.0                    // 固件位置环 P 增益（microduck 保留值）
    // duty = (q_target − q)·kp·error_gain；error_gain = 编码器/(KP_DIVISOR·PWM_LIMIT)
    static let errorGain = (4096.0 / (2 * Double.pi)) / (256.0 * 885.0)
    static let maxPWM = 1.0                    // duty 物理钳位；训练无 current limiter
    // 电压：训练 DR vin∈[6.5,8.2]、drop_gain∈[0,0.2]，彩排坍缩到 infer_policy CLI 默认
    static let vin = 7.4                       // 标称 2S LiPo
    static let vinDropGain = 0.1               // V_drop = gain·Σ|τ|（上一子步电机扭矩）
    static let vinMin = 6.0                    // 跌落后下限
    // m6 摩擦参数（directional + stribeck + quadratic）
    static let frictionBase = 0.004771183165566
    static let frictionStribeck = 0.004676345799486616
    static let lfMotor = 0.2667860954283698
    static let lfExternal = 8.515871897059342e-06
    static let lfMotorStribeck = 1.0722918395099123e-05
    static let lfExternalStribeck = 0.08077928978935671
    static let lfMotorQuad = 0.009972471242139415
    static let lfExternalQuad = 0.004902565732332559
    static let dthetaStribeck = 2.890372094130307
    static let stribeckAlpha = 8.683259907618984
    static let frictionViscous = 0.005359668274599504
    // 硬摩擦约束（bam.mjlab.BamActuator stiff_frictionloss；负 solref = direct 格式）
    static let stiffSolref: (Double, Double) = (-5.0e4, -2.0e2)
    static let stiffSolimp: (Double, Double, Double, Double, Double) = (0.99, 0.9999, 0.001, 0.5, 2.0)
    static var forceLimit: Double { vin * kt / R }   // ±0.9633 Nm
}

/// 模型改写：position 执行器 → BAM 扭矩电机（load_mujoco_with_bam /
/// BamActuator.edit_spec 的编译后等价物）。改动挂在共享 mjModel 上（模型级开关），
/// restore 可完整还原 XML PD 语义。
final class BamModelEdit {
    private let m: UnsafeMutablePointer<mjModel>
    private struct Snapshot {
        var gaintype: [Int32], biastype: [Int32], gainprm: [Double], biasprm: [Double]
        var ctrllimited: [Bool], forcelimited: [Bool], forcerange: [Double], gear: [Double]
        var damping: [Double], frictionloss: [Double], armature: [Double]
        var solref: [Double], solimp: [Double]
    }
    private var snapshot: Snapshot?
    private(set) var applied = false

    init(m: UnsafeMutablePointer<mjModel>) { self.m = m }

    /// 受控关节清单：执行器 i → (执行器下标, 关节 id, dof 下标)，跳过 passive_* 目标
    ///（与 infer_policy load_mujoco_with_bam 的 target 过滤一致）。
    static func actuatedJoints(m: UnsafeMutablePointer<mjModel>) -> [(act: Int, jnt: Int, dof: Int)] {
        let mm = m.pointee
        var out: [(Int, Int, Int)] = []
        for ai in 0..<Int(mm.nu) {
            let jnt = Int(mm.actuator_trnid[ai * 2])
            if let c = mj_id2name(m, Int32(mjOBJ_JOINT.rawValue), Int32(jnt)),
               String(cString: c).hasPrefix("passive_") { continue }
            out.append((ai, jnt, Int(mm.jnt_dofadr[jnt])))
        }
        return out
    }

    func apply(d: UnsafeMutablePointer<mjData>) {
        if applied { return }
        let mm = m.pointee
        let nu = Int(mm.nu), nv = Int(mm.nv)
        if snapshot == nil {
            func snapD(_ p: UnsafeMutablePointer<Double>, _ n: Int) -> [Double] { Array(UnsafeBufferPointer(start: p, count: n)) }
            func snapI(_ p: UnsafeMutablePointer<Int32>, _ n: Int) -> [Int32] { Array(UnsafeBufferPointer(start: p, count: n)) }
            func snapB(_ p: UnsafeMutablePointer<Bool>, _ n: Int) -> [Bool] { Array(UnsafeBufferPointer(start: p, count: n)) }
            snapshot = Snapshot(
                gaintype: snapI(mm.actuator_gaintype, nu), biastype: snapI(mm.actuator_biastype, nu),
                gainprm: snapD(mm.actuator_gainprm, nu * 10), biasprm: snapD(mm.actuator_biasprm, nu * 10),
                ctrllimited: snapB(mm.actuator_ctrllimited, nu), forcelimited: snapB(mm.actuator_forcelimited, nu),
                forcerange: snapD(mm.actuator_forcerange, nu * 2), gear: snapD(mm.actuator_gear, nu * 6),
                damping: snapD(mm.dof_damping, nv), frictionloss: snapD(mm.dof_frictionloss, nv),
                armature: snapD(mm.dof_armature, nv),
                solref: snapD(mm.dof_solref, nv * 2), solimp: snapD(mm.dof_solimp, nv * 5))
        }
        for (ai, _, dof) in Self.actuatedJoints(m: m) {
            // set_to_motor：固定增益 1、无偏置（biasprm 在 BIAS_NONE 下惰性，与 warp
            // edit_spec 一致——这里显式清零）、gear=1、电压界 forcerange
            mm.actuator_gaintype[ai] = Int32(mjGAIN_FIXED.rawValue)
            mm.actuator_biastype[ai] = Int32(mjBIAS_NONE.rawValue)
            mm.actuator_gainprm[ai * 10] = 1.0
            for k in 0..<10 { mm.actuator_biasprm[ai * 10 + k] = 0 }
            mm.actuator_gear[ai * 6] = 1.0
            for k in 1..<6 { mm.actuator_gear[ai * 6 + k] = 0 }
            mm.actuator_ctrllimited[ai] = false
            mm.actuator_forcelimited[ai] = true
            mm.actuator_forcerange[ai * 2] = -BamM6.forceLimit
            mm.actuator_forcerange[ai * 2 + 1] = BamM6.forceLimit
            // BAM 每子步重写摩擦/阻尼；关节侧先清零 + 硬摩擦约束 + 转子惯量
            mm.dof_damping[dof] = 0
            mm.dof_frictionloss[dof] = 0
            mm.dof_solref[dof * 2] = BamM6.stiffSolref.0
            mm.dof_solref[dof * 2 + 1] = BamM6.stiffSolref.1
            mm.dof_solimp[dof * 5] = BamM6.stiffSolimp.0
            mm.dof_solimp[dof * 5 + 1] = BamM6.stiffSolimp.1
            mm.dof_solimp[dof * 5 + 2] = BamM6.stiffSolimp.2
            mm.dof_solimp[dof * 5 + 3] = BamM6.stiffSolimp.3
            mm.dof_solimp[dof * 5 + 4] = BamM6.stiffSolimp.4
            mm.dof_armature[dof] = BamM6.armature
        }
        mj_setConst(m, d)
        applied = true
    }

    func restore(d: UnsafeMutablePointer<mjData>) {
        guard applied, let s = snapshot else { return }
        let mm = m.pointee
        let nu = Int(mm.nu), nv = Int(mm.nv)
        func putD(_ p: UnsafeMutablePointer<Double>, _ v: [Double]) { for i in 0..<v.count { p[i] = v[i] } }
        func putI(_ p: UnsafeMutablePointer<Int32>, _ v: [Int32]) { for i in 0..<v.count { p[i] = v[i] } }
        func putB(_ p: UnsafeMutablePointer<Bool>, _ v: [Bool]) { for i in 0..<v.count { p[i] = v[i] } }
        putI(mm.actuator_gaintype, s.gaintype); putI(mm.actuator_biastype, s.biastype)
        putD(mm.actuator_gainprm, s.gainprm); putD(mm.actuator_biasprm, s.biasprm)
        putB(mm.actuator_ctrllimited, s.ctrllimited); putB(mm.actuator_forcelimited, s.forcelimited)
        putD(mm.actuator_forcerange, s.forcerange); putD(mm.actuator_gear, s.gear)
        putD(mm.dof_damping, s.damping); putD(mm.dof_frictionloss, s.frictionloss)
        putD(mm.dof_armature, s.armature); putD(mm.dof_solref, s.solref); putD(mm.dof_solimp, s.solimp)
        _ = (nu, nv)
        mj_setConst(m, d)
        applied = false
    }
}

/// 每只鸭子一个实例（q_target / prevTorque 是逐实例状态；摩擦/阻尼写在共享模型上，
/// 因此多鸭子须按「每只 update 后立刻 mj_step」交错——main.swift 的 stepPhysics 已如此）。
final class BamController {
    private let m: UnsafeMutablePointer<mjModel>
    private let d: UnsafeMutablePointer<mjData>
    private var actIdx: [Int] = []     // ctrl 下标
    private var jntIdx: [Int] = []     // 关节 id（efc_id 匹配用，对齐 Python 参考）
    private var qposAdr: [Int] = []
    private var dofAdr: [Int] = []
    /// 位置目标 [rad]（策略拍写入；= DEFAULT_POSE + action）
    var qTarget: [Double]
    private var prevTorque: [Double]

    init?(m: UnsafeMutablePointer<mjModel>, d: UnsafeMutablePointer<mjData>) {
        self.m = m
        self.d = d
        let mm = m.pointee
        for (ai, jnt, dof) in BamModelEdit.actuatedJoints(m: m) {
            actIdx.append(ai)
            jntIdx.append(jnt)
            qposAdr.append(Int(mm.jnt_qposadr[jnt]))
            dofAdr.append(dof)
        }
        guard !actIdx.isEmpty else { print("bam: 无受控执行器"); return nil }
        qTarget = [Double](repeating: 0, count: actIdx.count)
        prevTorque = [Double](repeating: 0, count: actIdx.count)
    }

    /// bam_ctrl.reset(qpos)：清电压跌落状态，q_target = 当前关节位
    func reset() {
        for j in 0..<actIdx.count { qTarget[j] = d.pointee.qpos[qposAdr[j]] }
        prevTorque = [Double](repeating: 0, count: actIdx.count)
    }

    /// 200Hz 子步前调用（MujocoController.update 等价物）：
    /// duty→V→τ 写入 ctrl，摩擦预算写入 dof_frictionloss/damping（mj_step 的求解器
    /// 在下一步应用）。
    func update() {
        let mm = m.pointee
        let dd = d.pointee
        // 电压跌落：vin_eff = max(vin − gain·Σ|prevτ|, vin_min)
        var load = 0.0
        for t in prevTorque { load += abs(t) }
        let vinEff = max(BamM6.vin - BamM6.vinDropGain * load, BamM6.vinMin)
        // 固件 P 环（duty 钳 ±max_pwm）→ DC 电机方程（含反电动势，无 current limiter）
        for j in 0..<actIdx.count {
            let q = dd.qpos[qposAdr[j]]
            let dq = dd.qvel[dofAdr[j]]
            let duty = min(max((qTarget[j] - q) * BamM6.kpFW * BamM6.errorGain,
                               -BamM6.maxPWM), BamM6.maxPWM)
            let tau = BamM6.kt * (vinEff * duty) / BamM6.R
                - BamM6.kt * BamM6.kt * dq / BamM6.R
            dd.ctrl[actIdx[j]] = tau
            prevTorque[j] = tau
        }
        // 摩擦预算（m6：directional + stribeck + quadratic）
        var fricF = [Double](repeating: 0, count: actIdx.count)
        let frictionDof = Int32(mjCNSTR_FRICTION_DOF.rawValue)
        for e in 0..<Int(dd.nefc) where dd.efc_type[e] == frictionDof {
            let id = Int(dd.efc_id[e])
            for j in 0..<jntIdx.count where jntIdx[j] == id {
                fricF[j] += dd.efc_force[e]
                break
            }
        }
        for j in 0..<actIdx.count {
            let dof = dofAdr[j]
            let ext = -dd.qfrc_bias[dof] + dd.qfrc_constraint[dof] - fricF[j]
            let mot = dd.qfrc_actuator[dof]
            let dq = dd.qvel[dof]
            let gearbox = abs(ext * BamM6.lfExternal - mot * BamM6.lfMotor)
            let gearboxStr = abs(ext * BamM6.lfExternalStribeck - mot * BamM6.lfMotorStribeck)
            let sc = exp(-pow(abs(dq / BamM6.dthetaStribeck), BamM6.stribeckAlpha))
            var fl = BamM6.frictionBase + gearbox + sc * BamM6.frictionStribeck + gearboxStr * sc
            // quadratic 项：仅在外力与电机扭矩异号时启用
            let sgnE: Double = ext > 0 ? 1 : (ext < 0 ? -1 : 0)
            let sgnM: Double = mot > 0 ? 1 : (mot < 0 ? -1 : 0)
            if sgnE != sgnM {
                var quad = 0.0
                if abs(ext) < abs(mot) { quad += BamM6.lfExternalQuad * ext * ext }
                if abs(ext) > abs(mot) { quad += BamM6.lfMotorQuad * mot * mot }
                fl += sc * quad
            }
            mm.dof_frictionloss[dof] = fl
            mm.dof_damping[dof] = BamM6.frictionViscous
        }
    }
}
