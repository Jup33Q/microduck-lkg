import AppKit
import Foundation
import Metal
import QuartzCore
import simd
import MuJoCo
import StreamQuilt

// P1 viewer：microduck（或任意 MJCF）→ mjvScene → Metal 离轴 quilt。
// P2：app 模式接 StreamQuilt 显示侧（LKGApp：校准 → interlace → LKG 设备窗 +
// 主屏预览窗）。
//
//   swift run mujoco-lkg-viewer [scene.xml] [flags]
//     --layout 11x6|7x8   quilt 布局（默认 11x6 = LKG Go 主选）
//     --dump out.png      渲染一帧 quilt 存 PNG 后退出
//     --dump-lentic out.png  交织上屏图 dump（1440×2560，校准同 app 模式）
//     --from-hdr q.hdr    dump-lentic 的 quilt 输入改为 saveQuiltHDR 的
//                         rgba16Float raw + .json sidecar（固定输入离线回归）
//     --bench N           渲染 N 帧（每帧 mj_step）并报告 ms/frame
//     --steps N           dump/bench 前先 mj_step N 步（默认 0 = keyframe 站立姿态）
//     --half              quilt 半分辨率渲染（renderScale 0.5）
//     --sweep X --dist X --fov DEG --az DEG   相机覆盖（默认自动取景）
//     --test-pattern      app 模式启动即校准测试图（等价预览窗按 c）
//     --stream [port]     训练直播：不跑本地物理，跟随 UDP qpos 流渲染
//                         （microduck_rl lkg_stream，默认端口 44123）
//     --no-bam            策略驾驶退回 XML PD（默认 BAM M6 = infer_policy 默认路径）
//     --bam-trace out.csv [--trace-ticks N] [--trace-cmd x,y,z] [--policy p.onnx]
//                         BAM 离线对照：固定节拍逐子步写 CSV 后退出
//                         （与 scripts/bam_trace_ref.py 的 Python 参考 diff 验证）
//     --headcam-dump DIR --poses poses.json
//                         VLM 域差验证：按 poses.json 摆鸭子/球，渲 256×256
//                         head-cam（quackd render_headcam 语义：fovy 90、
//                         藏 group1/2、黑底、headlight 朴素光照）存
//                         DIR/viewer_<name>.png 后退出
//   无 --dump/--dump-lentic/--bench 时进入 app 模式：LKG 设备窗 60Hz 推流 +
//   主屏 tonemapped quilt 预览窗（含 tile 网格），每帧 mj_step 物理同进程推进。

setvbuf(stdout, nil, _IONBF, 0)

let defaultXML = AppRes.defaultScene   // P5：bundle Resources/竞技场 → 开发路径回退

var xmlPath = defaultXML
var layout = "7x8"   // 2026-10-07 定案：56 views（LKG Go 备选布局，顶点吞吐更省）；--layout 11x6 可切回
var dumpPath: String?
var dumpLenticPath: String?
var fromHDRPath: String?
var benchFrames = 0
var preSteps = 0
var renderScale: Float = 1
var startTestPattern = false
var streamPort: Int?
var policyPath: String?
var duckCountCLI = 1
// BAM M6 执行器：策略驾驶默认开（= infer_policy 默认路径，训练同款电压模型）；
// --no-bam 退回 XML PD。--bam-trace 为离线对照模式（写 CSV 后退出，不进 app）。
var bamEnabled = true
var bamTracePath: String?
var headDumpDir: String?
var posesPath: String?
var traceTicks = 200          // 策略拍数（200 拍 = 4s）
var traceCmd = "0.3,0,0"      // vx,vy,vyaw
var sweepOv: Float?; var distOv: Float?; var fovOv: Float?; var azOv: Float?
var diag = false
var vlmFlag = false   // --vlm：启动即策略驾驶 + VLM 高层（默认 alpha_walking，需竞技场场景）
var headCmdCLI: String?   // --head-cmd pitch,yaw（度）：bam-trace 下验证头部命令槽
// dist 在自动取景阶段才真正赋值；先给初值供 collectDrawItems（地面缩放）使用。
var dist: Float = 1
// 策略驾驶模式下地面跟随相机（鸭子会走出初始地面范围）
var floorFollow = false

let argv = CommandLine.arguments
var i = 1
while i < argv.count {
    switch argv[i] {
    case "--layout": layout = argv[i + 1]; i += 1
    case "--dump": dumpPath = argv[i + 1]; i += 1
    case "--dump-lentic": dumpLenticPath = argv[i + 1]; i += 1
    case "--from-hdr": fromHDRPath = argv[i + 1]; i += 1
    case "--test-pattern": startTestPattern = true
    case "--ducks": duckCountCLI = Int(argv[i + 1]) ?? 1; i += 1
    case "--stream":
        streamPort = 44123
        if i + 1 < argv.count, let p = Int(argv[i + 1]), !argv[i + 1].hasPrefix("--") {
            streamPort = p; i += 1
        }
    case "--policy":
        policyPath = "/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/microduck_rl/microduck_m1_800.onnx"
        if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
            policyPath = argv[i + 1]; i += 1
        }
    case "--no-bam": bamEnabled = false
    case "--bam-trace": bamTracePath = argv[i + 1]; i += 1
    case "--headcam-dump": headDumpDir = argv[i + 1]; i += 1
    case "--poses": posesPath = argv[i + 1]; i += 1
    case "--trace-ticks": traceTicks = Int(argv[i + 1]) ?? 200; i += 1
    case "--trace-cmd": traceCmd = argv[i + 1]; i += 1
    case "--bench": benchFrames = Int(argv[i + 1]) ?? 120; i += 1
    case "--steps": preSteps = Int(argv[i + 1]) ?? 0; i += 1
    case "--half": renderScale = 0.5
    case "--sweep": sweepOv = Float(argv[i + 1]); i += 1
    case "--dist": distOv = Float(argv[i + 1]); i += 1
    case "--fov": fovOv = Float(argv[i + 1]); i += 1
    case "--az": azOv = Float(argv[i + 1]); i += 1
    case "--diag": diag = true
    case "--vlm": vlmFlag = true
    case "--head-cmd": headCmdCLI = argv[i + 1]; i += 1
    default: if !argv[i].hasPrefix("--") { xmlPath = argv[i] }
    }
    i += 1
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(1)
}

// P5 会话恢复：无 CLI 参数（双击启动）时恢复上次会话配置；有任何参数则 CLI 全量主导
let cliArgs = CommandLine.arguments.filter { !$0.hasPrefix("-psn") }
if cliArgs.count == 1, let s = SessionConfig.load() {
    if let p = s.scenePath, FileManager.default.fileExists(atPath: p) { xmlPath = p }
    if let l = s.layout { layout = l }
    if let p = s.policyPath, FileManager.default.fileExists(atPath: p) { policyPath = p }
    if s.policyMode == true && policyPath == nil { policyPath = AppRes.defaultPolicy }
    if let b = s.bam { bamEnabled = b }
    if let n = s.ducks { duckCountCLI = n }
    if let v = s.fovDeg { fovOv = Float(v) }
    if let a = s.azDeg { azOv = Float(a) }
    if s.vlm == true { vlmFlag = true }   // sidecar 缺失时运行时再降级
} else if cliArgs.count == 1 && policyPath == nil {
    // 首次双击（无会话）：直接进策略驾驶（demo-first），面板可切自由物理；
    // sidecar 环境在位则连 VLM 一起开（自主找球踢球演示）
    if FileManager.default.fileExists(atPath: AppRes.defaultPolicy) { policyPath = AppRes.defaultPolicy }
    if LayaSidecar.available { vlmFlag = true }
}

// ---- 载入模型 ----------------------------------------------------------------
// 含 {{ROBOT_DIR}} 模板的场景（arena_ball.xml）先替换为解析出的机器人目录再加载
//（bundle Resources/robot 或开发路径），经临时文件喂 mj_loadXML。
var errBuf = [CChar](repeating: 0, count: 1000)
var xmlToLoad = xmlPath
if let raw = try? String(contentsOfFile: xmlPath, encoding: .utf8), raw.contains("{{ROBOT_DIR}}") {
    let cooked = raw.replacingOccurrences(of: "{{ROBOT_DIR}}", with: AppRes.robotDir)
    let tmp = NSTemporaryDirectory() + "mjlkg_scene_cooked.xml"
    try? cooked.write(toFile: tmp, atomically: true, encoding: .utf8)
    xmlToLoad = tmp
}
guard let m = mj_loadXML(xmlToLoad, nil, &errBuf, 1000) else {
    fail("mj_loadXML failed: \(String(cString: errBuf))")
}
guard let d = mj_makeData(m) else { fail("mj_makeData failed") }
let mm = m.pointee
print("=== model: \(xmlPath.split(separator: "/").last!) ngeom=\(mm.ngeom) nmesh=\(mm.nmesh) nmat=\(mm.nmat) ntex=\(mm.ntex)")
if mm.nkey > 0 { mj_resetDataKeyframe(m, d, 0) }
mj_forward(m, d)

let nq = Int(mm.nq)
func qposNorm() -> Double {
    var s = 0.0
    for k in 0..<nq { let v = d.pointee.qpos[k]; s += v * v }
    return s.squareRoot()
}

// 竞技场/VLM 部件（arena_ball.xml）：head_camera + 球 free joint
let headCamId = mj_name2id(m, Int32(mjOBJ_CAMERA.rawValue), "head_camera")
let ballJ = mj_name2id(m, Int32(mjOBJ_JOINT.rawValue), "ball_free")
let ballQ = ballJ >= 0 ? Int(m.pointee.jnt_qposadr[Int(ballJ)]) : -1
let ballV = ballJ >= 0 ? Int(m.pointee.jnt_dofadr[Int(ballJ)]) : -1
let arenaVLM = headCamId >= 0 && ballQ >= 0
let trunkBodyId = mj_name2id(m, Int32(mjOBJ_BODY.rawValue), "trunk_base")

// ---- --bam-trace：BAM 离线对照（headless，固定节拍跑策略，逐子步写 CSV 后退出）。
// 与 mujoco-lkg/scripts/bam_trace_ref.py（infer_policy BAM 参考）同一场景同一格式，
// 用 diff 做 bit 级移植验证。
if let tracePath = bamTracePath {
    let nv = Int(mm.nv), nu = Int(mm.nu)
    let onnx = policyPath
        ?? "/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg/models/pollen-policies/alpha_walking.onnx"
    let edit = BamModelEdit(m: m)
    if bamEnabled { edit.apply(d: d) }
    m.pointee.opt.timestep = 0.005   // infer_policy：200Hz 物理（同 setupDucks）
    guard let pol = PolicyRunner(m: m, d: d, onnxPath: onnx, useBam: bamEnabled) else {
        fail("bam-trace: policy 加载失败")
    }
    pol.resetPose()
    let cparts = traceCmd.split(separator: ",").map { Float($0) ?? 0 }
    pol.command.vx = cparts.count > 0 ? cparts[0] : 0
    pol.command.vy = cparts.count > 1 ? cparts[1] : 0
    pol.command.vyaw = cparts.count > 2 ? cparts[2] : 0
    // --head-cmd pitch,yaw（度，quackd look 语义：pitch 正=低头）：验证命令槽→颈部关节
    if let hc = headCmdCLI {
        let hp = hc.split(separator: ",").map { (Float($0) ?? 0) * .pi / 180 }
        pol.command.head = SIMD4(0, hp.count > 0 ? hp[0] : 0, hp.count > 1 ? hp[1] : 0, 0)
        print("--- head-cmd: pitch=\(hp.count > 0 ? hp[0] : 0) yaw=\(hp.count > 1 ? hp[1] : 0) rad")
    }
    // 摩擦列 = 受控 dof 的 dof_frictionloss（update 写入值，mj_step 不变）
    let fricDofs = BamModelEdit.actuatedJoints(m: m).map { $0.dof }
    var csv = "step,time"
        + (0..<nq).map { ",qpos\($0)" }.joined()
        + (0..<nv).map { ",qvel\($0)" }.joined()
        + (0..<nu).map { ",ctrl\($0)" }.joined()
        + fricDofs.indices.map { ",fric\($0)" }.joined()
        + "\n"
    var step = 0
    for _ in 0..<traceTicks {
        pol.tick(wallDt: 0.02)
        for _ in 0..<4 {
            pol.stepPhysics()
            var row = "\(step)," + String(format: "%.17g", d.pointee.time)
            for k in 0..<nq { row += String(format: ",%.17g", d.pointee.qpos[k]) }
            for k in 0..<nv { row += String(format: ",%.17g", d.pointee.qvel[k]) }
            for k in 0..<nu { row += String(format: ",%.17g", d.pointee.ctrl[k]) }
            for dof in fricDofs { row += String(format: ",%.17g", m.pointee.dof_frictionloss[dof]) }
            csv += row + "\n"
            step += 1
        }
    }
    do {
        try csv.write(toFile: tracePath, atomically: true, encoding: .utf8)
        print("=== bam-trace OK: \(tracePath) (\(step) substeps, bam=\(bamEnabled), cmd=\(traceCmd), policy=\(onnx.split(separator: "/").last!))")
    } catch { fail("bam-trace 写文件失败: \(error)") }
    mj_deleteData(d)
    mj_deleteModel(m)
    exit(0)
}

// 顶层全局状态是 main-actor 隔离的；以下闭包全部标 @MainActor 以匹配。

// ---- Metal / quilt 初始化 -----------------------------------------------------
let spec: QuiltSpec = layout == "7x8" ? .lkgGo56 : .lkgGo
// 无 dump/bench 诉求 = app 模式：LKGApp 壳自带校准（Bridge 127.0.0.1，失败回退
// lkgGoFallback 并打印警告）、LKG 设备窗 60Hz 推流与主屏预览窗；quilt 渲染器
// 直接复用 app.renderer，物理/光栅初始化不变。
let appMode = dumpPath == nil && dumpLenticPath == nil && benchFrames == 0 && headDumpDir == nil
let app: LKGApp?
let quilt: QuiltRenderer
do {
    if appMode {
        let a = try LKGApp(spec: spec, renderScale: renderScale)
        app = a
        quilt = a.renderer
    } else {
        app = nil
        quilt = try QuiltRenderer(spec: spec, renderScale: renderScale)
    }
} catch { fail("QuiltRenderer/LKGApp init failed: \(error)") }

let meshes = uploadMeshes(m: mm, device: quilt.device)
print("--- meshes uploaded: \(meshes.count) (\(meshes.reduce(0) { $0 + $1.smoothIndexCount / 3 }) tris)")

// ---- mjvScene -----------------------------------------------------------------
var scn = mjvScene()
mjv_defaultScene(&scn)
var vopt = mjvOption()
mjv_defaultOption(&vopt)
var vcam = mjvCamera()
mjv_defaultFreeCamera(m, &vcam)
mjv_makeScene(m, &scn, 20000)
mjv_updateScene(m, d, &vopt, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scn)

let planeType = Int32(mjGEOM_PLANE.rawValue)
let meshType = Int32(mjGEOM_MESH.rawValue)

/// 当前帧 scene geom → DrawItem 列表（mesh/平面；其余图元跳过并计数）。
/// 多鸭子：每只一个 mjvScene，平面只在第 0 只的场景里画一次。
@MainActor func collectDrawItems(from s: inout mjvScene, includePlane: Bool = true) -> (items: [DrawItem], skipped: [String: Int]) {
    var items: [DrawItem] = []
    items.reserveCapacity(Int(s.ngeom))
    var skipped = [String: Int]()
    for k in 0..<Int(s.ngeom) {
        let g = s.geoms[k]
        let p = floatsOf(g.pos, 3)
        let a = floatsOf(g.mat, 9)   // row-major 3x3
        let rot = float3x3(columns: (
            SIMD3(a[0], a[3], a[6]),
            SIMD3(a[1], a[4], a[7]),
            SIMD3(a[2], a[5], a[8])
        ))
        var model = mat4(rot, SIMD3(p[0], p[1], p[2]))
        let rgba = floatsOf(g.rgba, 4)
        var misc = SIMD4<Float>(g.specular, g.shininess, g.emission, 0)
        if g.type == meshType && g.dataid >= 0 {
            let meshid = Int(g.dataid) / 2
            let flat = (Int(g.dataid) & 1) == 1
            guard meshid < meshes.count else { continue }
            items.append(DrawItem(kind: flat ? .meshFlat(meshes[meshid]) : .meshSmooth(meshes[meshid]),
                                  model: model, rgba: SIMD4(rgba[0], rgba[1], rgba[2], rgba[3]),
                                  misc: misc))
        } else if g.type == planeType && includePlane {
            let sz = floatsOf(g.size, 3)
            // 地面至少延伸到相机后方 2×dist，保证前景不露底；
            // 流模式（训练 env 网格布点）与策略驾驶（鸭子会走远）地面都跟相机中心
            let followFloor = streamPort != nil || floorFollow
            let xmlHalf = max(sz[0], sz[1])
            let half = max(xmlHalf, 3 * dist, followFloor ? 64 : 0)
            // 棋盘格标记：1=蓝格 0.125m（默认地板）；2=灰格 0.1m（quackd floor_cam，
            // group 3）；3=蓝格 + 竞技场边界描边（group 1 地板且场景有球）。
            // geom group 从 mjModel 按 objid 反查
            let grp = (g.objtype == 5 && g.objid >= 0) ? Int(mm.geom_group[Int(g.objid)]) : 0
            misc.w = grp == 3 ? 2 : (grp == 1 && ballQ >= 0 ? 3 : 1)
            misc.x = 0   // 地面不要高光（反光在光场里很跳）
            if followFloor {
                let t = floatsOf(g.pos, 3)
                model = mat4(rot, SIMD3(center.x, center.y, t[2]))
            }
            model = model * Matrix4x4.scale(half)
            items.append(DrawItem(kind: .plane, model: model,
                                  rgba: SIMD4(rgba[0], rgba[1], rgba[2], rgba[3]),
                                  misc: misc))
        } else if g.type == Int32(mjGEOM_SPHERE.rawValue) {
            let r = floatsOf(g.size, 3)[0]
            items.append(DrawItem(kind: .sphere, model: model * Matrix4x4.scale(r),
                                  rgba: SIMD4(rgba[0], rgba[1], rgba[2], rgba[3]), misc: misc))
        } else if g.type == Int32(mjGEOM_BOX.rawValue) {
            let sz = floatsOf(g.size, 3)
            var sc = matrix_identity_float4x4
            sc.columns.0.x = sz[0]; sc.columns.1.y = sz[1]; sc.columns.2.z = sz[2]
            items.append(DrawItem(kind: .box, model: model * sc,
                                  rgba: SIMD4(rgba[0], rgba[1], rgba[2], rgba[3]), misc: misc))
        } else {
            skipped["type\(g.type)", default: 0] += 1
        }
    }
    return (items, skipped)
}

var (items0, skipped0) = collectDrawItems(from: &scn)
print("--- scene geoms: \(scn.ngeom) → drawable \(items0.count)" +
      (skipped0.isEmpty ? "" : " skipped: \(skipped0)"))

// ---- --diag：scene pose vs geom_xpos，及 mesh frame 非恒等统计 -----------------
if diag {
    print("--- diag: scene pose vs d.geom_xpos/xmat + mesh frame")
    var nNonIdent = 0
    for k in 0..<Int(scn.ngeom) {
        let g = scn.geoms[k]
        if g.type != meshType || g.objid < 0 { continue }
        let oid = Int(g.objid)
        let sp = floatsOf(g.pos, 3)
        let dp = (0..<3).reduce(Float(0)) { max($0, abs(sp[$1] - Float(d.pointee.geom_xpos[oid * 3 + $1]))) }
        let sm = floatsOf(g.mat, 9)
        let dm = (0..<9).reduce(Float(0)) { max($0, abs(sm[$1] - Float(d.pointee.geom_xmat[oid * 9 + $1]))) }
        let mid = Int(g.dataid) / 2
        let mp = (0..<3).map { mm.mesh_pos[mid * 3 + $0] }
        let mq = (0..<4).map { mm.mesh_quat[mid * 4 + $0] }
        let ms = (0..<3).map { mm.mesh_scale[mid * 3 + $0] }
        let offNI = mp.reduce(0) { $0 + abs($1) } + abs(mq[0] - 1) + mq[1...].reduce(0) { $0 + abs($1) }
        let sclNI = ms.reduce(0) { $0 + abs($1 - 1) }
        if offNI > 1e-6 || sclNI > 1e-6 { nNonIdent += 1 }
        if dp > 1e-5 || dm > 1e-5 || offNI > 1e-4 || sclNI > 1e-4 {
            print(String(format: "  objid=%2d mesh=%2d |dpos|=%.6f |dmat|=%.6f meshpos=(%.4f %.4f %.4f) quat=(%.3f %.3f %.3f %.3f) scale=(%.4f %.4f %.4f)",
                         oid, mid, dp, dm, mp[0], mp[1], mp[2], mq[0], mq[1], mq[2], mq[3], ms[0], ms[1], ms[2]))
        }
    }
    print("--- diag: meshes with non-identity frame: \(nNonIdent)/70")
}

// ---- 自动取景：非平面 geom 的 rbound bbox 定主体 -------------------------------
var lo = SIMD3<Float>(repeating: .infinity)
var hi = SIMD3<Float>(repeating: -.infinity)
for k in 0..<Int(scn.ngeom) {
    let g = scn.geoms[k]
    if g.type == planeType { continue }
    // 竞技场模式：取景只对机器人（mesh geom）——墙/球是场景设施，不参与主体 bbox
    if arenaVLM && g.type != meshType { continue }
    let p = floatsOf(g.pos, 3)
    let c = SIMD3(p[0], p[1], p[2])
    let r = g.modelrbound > 0 ? g.modelrbound : 0
    lo = min(lo, c - r); hi = max(hi, c + r)
}
let center0 = (lo + hi) * 0.5
var center = center0   // 流模式下跟随训练 env 的 base xy（env origins 是网格布点）
let half = (hi - lo) * 0.5
let radius = length(half)

var camera = QuiltCamera()
let renderer = try MjvQuiltRenderer(quilt: quilt, camera: camera)
// 相机默认 fov=45°（P2 实机目检定案，25° 画面偏扁）；az=205° 为 microduck 正面 3/4。
let initFovDeg = fovOv ?? 45
let initAzDeg = azOv ?? 205   // microduck 前脸朝 +x 偏 -y

/// 重建取景：fov/az/距离倍率 + 相机帧目标偏移（控制面板实时调用）。
/// 相机水平无俯仰（P1 定案：俯仰破坏焦点平面跨视角对齐），上下构图靠 offsetY 平移。
@MainActor func applyFraming(fovDeg: Float, azDeg: Float, margin: Float,
                             offsetX: Float = 0, offsetY: Float = 0) {
    camera.fovY = fovDeg * .pi / 180
    let tanF = tan(camera.fovY * 0.5)
    // 分轴贴合：相机帧里水平半宽 = half.x、竖直半高 = half.z（mujoco z-up → 相机 y-up）
    dist = max(half.z / tanF, half.x / (tanF * spec.tileAspect)) * margin
    if let d0 = distOv { dist = d0 }
    camera.distance = dist
    camera.sweep = sweepOv ?? max(2 * half.x, 0.8 * radius)
    renderer.camera = camera
    let azr = azDeg * .pi / 180
    // 相机帧：MuJoCo z-up → y-up，主体中心移到原点（焦点平面 z=0），az 绕世界 z 摆正朝向；
    // offset 以 radius 为单位在相机帧内平移主体（正 offsetY = 主体上移）
    renderer.worldFromMujoco = Matrix4x4.translation(SIMD3(offsetX * radius, offsetY * radius, 0))
        * Matrix4x4.rotationX(-.pi / 2) * rotationZ(azr) * Matrix4x4.translation(-center)
    renderer.nearPlane = max(0.005, dist - 4 * radius)
    renderer.farPlane = dist + 8 * radius
}
applyFraming(fovDeg: initFovDeg, azDeg: initAzDeg, margin: 1.32)
print(String(format: "--- framing: center=(%.3f %.3f %.3f) r=%.3f sweep=%.3f dist=%.3f fov=%.0f° az=%.0f° near=%.3f far=%.2f",
             center.x, center.y, center.z, radius, camera.sweep, dist,
             camera.fovY * 180 / .pi, initAzDeg, renderer.nearPlane, renderer.farPlane))
print(String(format: "--- focus-plane z=0 pixel check: max cross-view deviation = %.5f px", renderer.focusPlanePixelError()))

// ---- 渲染一帧（mjvScene 同步 → encode） ----------------------------------------
var frameMs = 0.0
var gpuMs = 0.0
@MainActor func renderFrame() {
    mjv_updateScene(m, d, &vopt, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scn)
    let (items, _) = collectDrawItems(from: &scn)
    guard let cmd = quilt.commandQueue.makeCommandBuffer() else { return }
    let t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    renderer.encodeScene(cmd: cmd, items: items)
    cmd.commit()
    cmd.waitUntilCompleted()
    let t1 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    frameMs = Double(t1 - t0) / 1e6
    if cmd.gpuStartTime > 0 { gpuMs = (cmd.gpuEndTime - cmd.gpuStartTime) * 1000 }
}

/// saveQuiltHDR 的 rgba16Float raw + .json sidecar → 共享 MTLTexture
///（跨引擎 interlace 离线回归的固定 quilt 输入）。
func loadHDRQuilt(path: String, device: MTLDevice) -> MTLTexture? {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url),
          let mj = try? Data(contentsOf: url.deletingPathExtension().appendingPathExtension("json")),
          let meta = try? JSONSerialization.jsonObject(with: mj) as? [String: Any],
          let w = meta["width"] as? Int, let h = meta["height"] as? Int,
          data.count == w * h * 8 else { return nil }
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                      width: w, height: h, mipmapped: false)
    td.usage = [.shaderRead]
    td.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: td) else { return nil }
    data.withUnsafeBytes { ptr in
        tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                    withBytes: ptr.baseAddress!, bytesPerRow: w * 8)
    }
    return tex
}

// ---- 模式分发 ------------------------------------------------------------------
for _ in 0..<preSteps { mj_step(m, d) }
mj_forward(m, d)

if let dumpPath {
    renderFrame()
    print(String(format: "--- dump frame: wall=%.2fms gpu=%.2fms draws=%d tris=%d",
                 frameMs, gpuMs, renderer.drawCalls, renderer.triangles))
    let (items, _) = collectDrawItems(from: &scn)
    quilt.saveQuiltPNG(to: dumpPath) { cmd in
        renderer.encodeScene(cmd: cmd, items: items)
    }
    print("=== dump OK: \(dumpPath)")
}

if let dumpLenticPath {
    // 与 sq-demo --dump-lentic 同路径：fetchFromBridge（127.0.0.1），失败回退
    // lkgGoFallback。比对时两边校准来源必须一致（serial 打印出来供核对）。
    let calibration = Calibration.fetchFromBridge() ?? .lkgGoFallback
    print("--- dump-lentic calibration: \(calibration.serial)")
    if let fromHDRPath {
        guard let tex = loadHDRQuilt(path: fromHDRPath, device: quilt.device) else {
            fail("load --from-hdr failed: \(fromHDRPath)（需要 raw + .json sidecar）")
        }
        guard tex.width == spec.width && tex.height == spec.height else {
            fail("--from-hdr \(tex.width)x\(tex.height) 与 spec \(spec.width)x\(spec.height) 不符（须同布局全分辨率）")
        }
        quilt.saveLenticularPNG(to: dumpLenticPath, calibration: calibration, source: tex) { _ in }
    } else {
        quilt.saveLenticularPNG(to: dumpLenticPath, calibration: calibration) { cmd in
            let (items, _) = collectDrawItems(from: &scn)
            renderer.encodeScene(cmd: cmd, items: items)
        }
    }
    print("=== dump-lentic OK: \(dumpLenticPath)")
}

/// head_camera 的针孔 viewProj（fovy 90°、aspect 1，quackd render_headcam 语义）
/// + eye + fwd。forward = cam_xmat 第 2 列（上游相机 quat 非 MuJoCo 视见约定，
/// +z 列才是镜头朝向——quackd 注释）。返回 MuJoCo 世界系矩阵（不套 worldFromMujoco）。
func headCamViewProj(d: UnsafeMutablePointer<mjData>, camId: Int32) -> (float4x4, SIMD3<Float>, SIMD3<Float>) {
    let ci = Int(camId)
    let eye = SIMD3<Float>(Float(d.pointee.cam_xpos[ci * 3]),
                           Float(d.pointee.cam_xpos[ci * 3 + 1]),
                           Float(d.pointee.cam_xpos[ci * 3 + 2]))
    let fwd = SIMD3<Float>(Float(d.pointee.cam_xmat[ci * 9 + 2]),
                           Float(d.pointee.cam_xmat[ci * 9 + 5]),
                           Float(d.pointee.cam_xmat[ci * 9 + 8]))
    let zc = -normalize(fwd)
    let xc = normalize(cross(SIMD3<Float>(0, 0, 1), zc))
    let yc = cross(zc, xc)
    let view = float4x4(columns: (
        SIMD4(xc.x, yc.x, zc.x, 0), SIMD4(xc.y, yc.y, zc.y, 0), SIMD4(xc.z, yc.z, zc.z, 0),
        SIMD4(-dot(xc, eye), -dot(yc, eye), -dot(zc, eye), 1)))
    let nclip: Float = 0.005, fclip: Float = 10
    let proj = float4x4(columns: (   // fovy 90° → f = 1/tan(45°) = 1
        SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0),
        SIMD4(0, 0, fclip / (nclip - fclip), -1),
        SIMD4(0, 0, nclip * fclip / (nclip - fclip), 0)))
    return (proj * view, eye, fwd)
}

/// head-cam 的躯干锁定变体（**VLM 输入专用**）：eye = 真实 head_camera 位置，
/// 朝向 = 躯干当前旋转 × reset 时刻的 head相机相对躯干姿态（home 位）——
/// 即「脖子焊死在 home」的虚拟相机。quackd 训练帧 = 脖子在 home 的物理头相机
/// （含步态摇摆，由 R_trunk 实时带进来），所以本变体对 VLM 是分布内的；
/// 物理脖子（注视反射/手动滑杆）怎么动都不影响 VLM 输入。
/// headHomeRel = R_trunkᵀ·R_cam @ reset（颈部 home）。
func headCamViewProjTrunkLocked(d: UnsafeMutablePointer<mjData>, camId: Int32,
                                trunkId: Int32, rel: float3x3) -> (float4x4, SIMD3<Float>, SIMD3<Float>) {
    let ci = Int(camId), bi = Int(trunkId)
    let eye = SIMD3<Float>(Float(d.pointee.cam_xpos[ci * 3]),
                           Float(d.pointee.cam_xpos[ci * 3 + 1]),
                           Float(d.pointee.cam_xpos[ci * 3 + 2]))
    func rotOf(_ p: UnsafeMutablePointer<Double>, _ o: Int) -> float3x3 {
        float3x3(columns: (SIMD3(Float(p[o]), Float(p[o + 3]), Float(p[o + 6])),
                           SIMD3(Float(p[o + 1]), Float(p[o + 4]), Float(p[o + 7])),
                           SIMD3(Float(p[o + 2]), Float(p[o + 5]), Float(p[o + 8]))))
    }
    let rTrunk = rotOf(d.pointee.xmat, bi * 9)
    let fwd = (rTrunk * rel).columns.2
    let zc = -normalize(fwd)
    let xc = normalize(cross(SIMD3<Float>(0, 0, 1), zc))
    let yc = cross(zc, xc)
    let view = float4x4(columns: (
        SIMD4(xc.x, yc.x, zc.x, 0), SIMD4(xc.y, yc.y, zc.y, 0), SIMD4(xc.z, yc.z, zc.z, 0),
        SIMD4(-dot(xc, eye), -dot(yc, eye), -dot(zc, eye), 1)))
    let nclip: Float = 0.005, fclip: Float = 10
    let proj = float4x4(columns: (
        SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0),
        SIMD4(0, 0, fclip / (nclip - fclip), -1),
        SIMD4(0, 0, nclip * fclip / (nclip - fclip), 0)))
    return (proj * view, eye, fwd)
}

// ---- --headcam-dump：VLM 域差验证（quackd render_headcam 语义的 Metal 复刻） ------
// 逐 poses.json state 摆位 → head_camera（cam_xpos/cam_xmat，forward=col2，fovy 90）
// → 256×256 rgba8 → PNG。与 quackd_*.png 同名配对，供 laya 推理分布对比。
if let headDumpDir {
    guard let posesPath,
          let pj = try? Data(contentsOf: URL(fileURLWithPath: posesPath)),
          let poseList = try? JSONSerialization.jsonObject(with: pj) as? [[String: Any]]
    else { fail("--headcam-dump 需要 --poses poses.json（gen_quackd_frames.py 生成）") }
    guard headCamId >= 0 else { fail("模型无 head_camera") }
    // head-cam 视角：group 0（碰撞/球/墙）+ group 3（碰撞体 + floor_cam 灰格）
    // + group 4（围墙——quilt 不画）；藏 group 1（蓝格地板）与 group 2（外壳 mesh）
    var voptHead = mjvOption()
    mjv_defaultOption(&voptHead)
    voptHead.geomgroup.1 = 0
    voptHead.geomgroup.2 = 0
    voptHead.geomgroup.3 = 1
    voptHead.geomgroup.4 = 1
    var scnHead = mjvScene()
    mjv_defaultScene(&scnHead)
    mjv_makeScene(m, &scnHead, 20000)
    let nvAll = Int(mm.nv)
    try? FileManager.default.createDirectory(atPath: headDumpDir, withIntermediateDirectories: true)
    for entry in poseList {
        guard let name = entry["name"] as? String,
              let duck = entry["duck"] as? [Double],
              let ball = entry["ball"] as? [Double] else { continue }
        // 鸭子：free joint (x, y, 0.125, yaw) + 14 关节 DEFAULT_POSE（= quackd reset）
        for k in 0..<Int(mm.nq) { d.pointee.qpos[k] = 0 }
        for k in 0..<nvAll { d.pointee.qvel[k] = 0 }
        d.pointee.qpos[0] = duck[0]
        d.pointee.qpos[1] = duck[1]
        d.pointee.qpos[2] = 0.125   // STAND_HEIGHT
        d.pointee.qpos[3] = cos(duck[2] / 2)
        d.pointee.qpos[6] = sin(duck[2] / 2)
        for k in 0..<14 { d.pointee.qpos[7 + k] = Double(PolicyRunner.defaultPose[k]) }
        if ballQ >= 0 {
            d.pointee.qpos[ballQ] = ball[0]
            d.pointee.qpos[ballQ + 1] = ball[1]
            d.pointee.qpos[ballQ + 2] = 0.05
            d.pointee.qpos[ballQ + 3] = 1
            for k in 0..<6 { d.pointee.qvel[ballV + k] = 0 }
        }
        mj_forward(m, d)
        // head_camera：eye=cam_xpos，forward=cam_xmat 第 2 列（quackd 语义），fovy 90
        let (headVP, eye, fwd) = headCamViewProj(d: d, camId: headCamId)
        mjv_updateScene(m, d, &voptHead, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scnHead)
        let (hItems, hSkipped) = collectDrawItems(from: &scnHead)
        guard let cmd = quilt.commandQueue.makeCommandBuffer() else { continue }
        renderer.encodeHeadCamera(cmd: cmd, items: hItems, viewProj: headVP, camPos: eye, camFwd: fwd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let pix = renderer.readHeadPixels()
        guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 256, pixelsHigh: 256,
                                         bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                         isPlanar: false, colorSpaceName: .deviceRGB,
                                         bytesPerRow: 256 * 4, bitsPerPixel: 32) else { continue }
        pix.withUnsafeBytes { memcpy(rep.bitmapData!, $0.baseAddress!, pix.count) }
        guard let png = rep.representation(using: .png, properties: [:]) else { continue }
        let out = "\(headDumpDir)/viewer_\(name).png"
        do { try png.write(to: URL(fileURLWithPath: out)) } catch { fail("写 \(out) 失败: \(error)") }
        let hp = entry["head_pose"] as? [Double]
        print(String(format: "--- headcam %@: eye=(%.4f %.4f %.4f) fwd=(%.3f %.3f %.3f)%@%@",
                     name, eye.x, eye.y, eye.z, fwd.x, fwd.y, fwd.z,
                     hp.map { String(format: " quackd=(%.4f %.4f %.4f yaw %.1f°)",
                                     $0[0], $0[1], $0[2], $0[3] * 180 / .pi) } ?? "",
                     hSkipped.isEmpty ? "" : " skipped: \(hSkipped)"))
    }
    print("=== headcam-dump OK: \(poseList.count) states → \(headDumpDir)/viewer_*.png")
    mjv_freeScene(&scnHead)
    mj_deleteData(d)
    mj_deleteModel(m)
    exit(0)
}

if benchFrames > 0 {
    let q0 = qposNorm()
    var wall = [Double](); wall.reserveCapacity(benchFrames)
    var gpu = [Double](); gpu.reserveCapacity(benchFrames)
    for _ in 0..<benchFrames {
        mj_step(m, d)
        renderFrame()
        wall.append(frameMs); gpu.append(gpuMs)
    }
    wall.sort(); gpu.sort()
    func stats(_ a: [Double]) -> String {
        String(format: "mean=%.2f p50=%.2f max=%.2f", a.reduce(0, +) / Double(a.count), a[a.count / 2], a.last!)
    }
    print("=== bench \(layout) (\(spec.columns)x\(spec.rows)=\(spec.viewCount) views, scale=\(renderScale)) x\(benchFrames) frames")
    print("    wall ms/frame: \(stats(wall))  → max \(String(format: "%.1f", 1000 / (wall.reduce(0, +) / Double(benchFrames)))) FPS")
    print("    gpu  ms/frame: \(stats(gpu))")
    print("    draws/frame=\(renderer.drawCalls) tris/frame=\(renderer.triangles)")
    print(String(format: "    physics in-process: |qpos| %.4f → %.4f (sim t=%.2fs)", q0, qposNorm(), d.pointee.time))
}

// ---- app 模式：LKGApp 壳（LKG 设备窗 60Hz + 主屏预览窗 + SwiftUI 控制面板） -----
if let app {
    // --stream [port]：训练直播模式。不跑本地 mj_step，跟随 UDP 推来的 qpos
    //（microduck_rl lkg_stream，MICRODUCK_LKG_STREAM=<port>）。
    let udp = UdpQposReceiver()
    if let streamPort {
        if let err = udp.start(port: UInt16(streamPort)) {
            fail("stream receiver: \(err)")
        }
        print("--- stream mode: listening udp :\(streamPort) for training qpos (nq=\(nq))")
    }
    if vlmFlag && policyPath == nil {
        policyPath = AppRes.defaultPolicy
    }
    let session = cliArgs.count == 1 ? SessionConfig.load() : nil
    let control = ViewerControl(app: app, fovDeg: Double(initFovDeg),
                                azDeg: Double(initAzDeg), margin: 1.32,
                                driveMode: policyPath != nil ? .policy : .free,
                                policyPath: policyPath ?? ViewerControl.defaultPolicyPath)
    control.duckCount = duckCountCLI
    control.actuatorBAM = bamEnabled
    control.vlmArenaOK = arenaVLM
    if let session {
        if let v = session.margin { control.margin = v }
        if let v = session.followSmooth { control.followSmooth = v }
        if let v = session.followHeading { control.followHeading = v }
        if let v = session.headPitchDeg { control.headPitchDeg = v }
        if let v = session.headYawDeg { control.headYawDeg = v }
    }
    if vlmFlag {
        control.vlmDrive = LayaSidecar.available
        if !LayaSidecar.available { print("--- vlm: sidecar 环境缺失（~/kimi-work/venvs/laya），VLM 关闭") }
        if !arenaVLM { print("--- vlm: 场景无 head_camera/ball_free，VLM 驾驶不可用（用竞技场场景）") }
    }
    app.onWillTerminate = {
        SessionConfig(scenePath: xmlPath, layout: layout,
                      policyMode: control.driveMode == .policy,
                      policyPath: control.policyPath, bam: control.actuatorBAM,
                      ducks: control.duckCount, vlm: control.vlmDrive,
                      fovDeg: control.fovDeg, azDeg: control.azDeg, margin: control.margin,
                      followSmooth: control.followSmooth, followHeading: control.followHeading,
                      headPitchDeg: control.headPitchDeg, headYawDeg: control.headYawDeg).save()
    }
    app.testPattern = startTestPattern
    app.showTileGrid = true
    app.onKeyPreemptsBuiltins = true   // WASD 的 s 与 LKGApp 内置存图键冲突，驾驶键优先
    let origTimestep = m.pointee.opt.timestep
    // ---- 多鸭子：N ×（mjData + PolicyRunner + mjvScene），指令广播 ----
    var datas: [UnsafeMutablePointer<mjData>] = [d]
    var scns: [mjvScene] = [scn]
    var policies: [PolicyRunner] = []
    let bamEdit = BamModelEdit(m: m)   // BAM 模型改写挂在共享 mjModel 上（可还原）
    var appliedDriveVersion: UInt = .max   // 强制首帧应用一次模式
    var appliedDuckVersion: UInt = .max
    var appliedPolicyVersion: UInt = 0
    /// VLM 躯干锁定虚拟相机的 home 基座（R_trunkᵀ·R_cam @ reset；setupDucks 时捕获）
    var headHomeRel: float3x3?
    /// 球是否已布过（策略接管/换策略不重布；仅首次进入/重置按钮/贴墙静止才布）
    var ballEverPlaced = false

    @MainActor func setupDucks(_ n: Int, respawnBall: Bool = false) {
        while datas.count > n && datas.count > 1 {
            let dd = datas.removeLast()
            mjv_freeScene(&scns[scns.count - 1])
            scns.removeLast()
            mj_deleteData(dd)
        }
        while datas.count < n {
            guard let nd = mj_makeData(m) else { break }
            var ns = mjvScene()
            mjv_defaultScene(&ns)
            mjv_makeScene(m, &ns, 20000)
            datas.append(nd)
            scns.append(ns)
        }
        if control.actuatorBAM { bamEdit.apply(d: datas[0]) } else { bamEdit.restore(d: datas[0]) }
        policies = datas.compactMap { PolicyRunner(m: m, d: $0, onnxPath: control.policyPath,
                                                   useBam: control.actuatorBAM) }
        // 原地复位（用户要求：策略接管不把鸭子搬回原点）：保留当前 xy+yaw，
        // 站立姿态原地重置；倒地 = quackd stand_up 语义（原地扶起）。
        // 多鸭子：鸭 0 保持原位，其余以鸭 0 为锚按网格补位。
        let anchorX = datas[0].pointee.qpos[0], anchorY = datas[0].pointee.qpos[1]
        for (i, pol) in policies.enumerated() {
            let dd = datas[i]
            let q0 = dd.pointee.qpos[0], q1 = dd.pointee.qpos[1]
            let qw = dd.pointee.qpos[3], qx = dd.pointee.qpos[4],
                qy = dd.pointee.qpos[5], qz = dd.pointee.qpos[6]
            // 躯干大致直立时才保留 yaw（倒地时 yaw 退化，回 0）
            let upright = dd.pointee.qpos[2] > 0.05
            let yaw = upright ? atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) : 0
            pol.resetPose()
            if i == 0 {
                dd.pointee.qpos[0] = q0
                dd.pointee.qpos[1] = q1
            } else {
                let col = i % 4, rowi = i / 4
                dd.pointee.qpos[0] = anchorX + Double((Float(col) - Float(min(n, 4) - 1) * 0.5) * 0.55)
                dd.pointee.qpos[1] = anchorY + Double((Float(rowi) - Float((n - 1) / 4) * 0.5) * 0.55)
            }
            dd.pointee.qpos[3] = cos(yaw / 2)
            dd.pointee.qpos[4] = 0
            dd.pointee.qpos[5] = 0
            dd.pointee.qpos[6] = sin(yaw / 2)
            mj_forward(m, dd)
        }
        // 竞技场布球：仅首次/显式重置时（接管与换策略不动球）
        if ballQ >= 0 && (respawnBall || !ballEverPlaced) {
            placeBall()
            ballEverPlaced = true
        }
        m.pointee.opt.timestep = 0.005   // infer_policy：200Hz 物理
        // 竞技场有界（围墙），地板固定不跟随；非竞技场策略驾驶地面跟相机
        floorFollow = ballQ < 0
        // VLM 躯干锁定相机基座：颈部此刻在 home（resetPose 刚写过）
        if arenaVLM && trunkBodyId >= 0 {
            let dd = datas[0]
            func rot3(_ p: UnsafeMutablePointer<Double>, _ o: Int) -> float3x3 {
                float3x3(columns: (SIMD3(Float(p[o]), Float(p[o + 3]), Float(p[o + 6])),
                                   SIMD3(Float(p[o + 1]), Float(p[o + 4]), Float(p[o + 7])),
                                   SIMD3(Float(p[o + 2]), Float(p[o + 5]), Float(p[o + 8]))))
            }
            headHomeRel = rot3(dd.pointee.xmat, Int(trunkBodyId) * 9).transpose
                * rot3(dd.pointee.cam_xmat, Int(headCamId) * 9)
        }
        yawOffsetSmooth = 0
        center = center0
        print("--- policy mode: \(policies.count) ducks × \(control.policyPath.split(separator: "/").last!)（\(control.actuatorBAM ? "BAM M6 / 训练同款" : "XML PD / --no-bam 语义")）")
    }
    var lastFrameT = CACurrentMediaTime()
    var yawOffsetSmooth: Float = 0   // 朝向跟随平滑偏置（度）
    var appliedCamVersion = control.cameraVersion
    var appliedReset = control.resetRequest
    var appliedPush: UInt = 0
    var ctlFrames = 0
    var ctlLastReport = CACurrentMediaTime()
    var lastPacketCount: UInt64 = 0
    var streamMismatchWarned = false
    // ---- VLM 驾驶（laya-vision-microduck-kick，竞技场场景限定） ----
    var voptHead = mjvOption()
    mjv_defaultOption(&voptHead)
    voptHead.geomgroup.1 = 0   // 藏蓝格地板
    voptHead.geomgroup.2 = 0   // 藏机器人外壳 mesh
    voptHead.geomgroup.3 = 1   // 显碰撞体 + floor_cam 灰格（quackd 头视角）
    voptHead.geomgroup.4 = 1   // 显围墙（group 4 = 仅 head-cam 可见，quilt 不画）
    var scnHead = mjvScene()
    mjv_defaultScene(&scnHead)
    if arenaVLM { mjv_makeScene(m, &scnHead, 20000) }
    /// 布球（quackd 分布：[-0.75,0.75]² 均匀、距鸭 0 ≥0.5m；所有 duck data 同步）
    @MainActor func placeBall() {
        guard ballQ >= 0 else { return }
        var bx = 0.0, by = 0.0
        for _ in 0..<1000 {
            bx = .random(in: -0.75...0.75)
            by = .random(in: -0.75...0.75)
            let dx = bx - datas[0].pointee.qpos[0], dy = by - datas[0].pointee.qpos[1]
            if dx * dx + dy * dy >= 0.25 { break }
        }
        for dd in datas {
            dd.pointee.qpos[ballQ] = bx
            dd.pointee.qpos[ballQ + 1] = by
            dd.pointee.qpos[ballQ + 2] = 0.05
            dd.pointee.qpos[ballQ + 3] = 1
            for k in 4..<7 { dd.pointee.qpos[ballQ + k] = 0 }
            for k in 0..<6 { dd.pointee.qvel[ballV + k] = 0 }
            mj_forward(m, dd)
        }
        print(String(format: "--- arena: ball @ (%.2f, %.2f)", bx, by))
    }
    var ballWallRestT = 0.0   // 球贴墙静止计时（0.8s 后重布）
    var vlmDbgLast = 0.0
    let vlm = LayaSidecar()
    var vlmCmd: (vx: Float, vyaw: Float) = (0, 0)   // 当前动作指令（到期清零）
    var vlmActionUntil = 0.0
    vlm.onLog = { s in print(s) }
    vlm.onDecision = { dec in
        let now = CACurrentMediaTime()
        control.vlmStatus = String(format: "%@  F=%.2f L=%.2f R=%.2f K=%.2f  %dms",
                                   dec.choice, dec.probs.f, dec.probs.l, dec.probs.r,
                                   dec.probs.k, dec.ms)// 球相对头相机的距离/方位角（决策与控制共用；ground truth 仅用于控制量，
        // 方向决策完全由 VLM 帧推理给出）
        let dd = datas[0]
        let ci = Int(headCamId)
        let hx = Float(dd.pointee.cam_xpos[ci * 3]), hy = Float(dd.pointee.cam_xpos[ci * 3 + 1])
        let bx = Float(dd.pointee.qpos[ballQ]), by = Float(dd.pointee.qpos[ballQ + 1])
        let qw = Float(dd.pointee.qpos[3]), qx = Float(dd.pointee.qpos[4]),
            qy = Float(dd.pointee.qpos[5]), qz = Float(dd.pointee.qpos[6])
        let yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        let ballDist = hypotf(bx - hx, by - hy)
        var ballBearing = atan2(by - hy, bx - hx) - yaw
        while ballBearing > .pi { ballBearing -= 2 * .pi }
        while ballBearing < -.pi { ballBearing += 2 * .pi }
        // quackd eval 语义：FORWARD=0.3m/s×1.0s（achieved ~0.11 ≈ 11cm/拍），
        // KICK 脚本推球。转向 cmd ±1.5 rad/s（BAM 下 1.0 不转，实测 1.5→achieved 0.7）。
        // **转向量受控**：时长 = 0.85×|bearing|/achieved（欠转 15% 防过冲），
        // clamp [0.08, 0.5]s——固定时长会过冲导致 VLM 左右震荡（1.5-0--1.5-0 抽动）。
        // **球不在视野时的扫描纠偏**（用户反馈：VLM 沿用 teacher 的「不可见就 LEFT」
        // 会原地无限转圈）：|bearing|>60° 时转向改走真值短边方向，让扫描收敛；
        // 球在视野内时方向完全由 VLM 决定（感知正确性仍由 VLM 负责）。
        let ballInView = abs(ballBearing) <= 60 * Float.pi / 180
        print(String(format: "vlm: %@ F=%.2f L=%.2f R=%.2f K=%.2f | ball dist=%.2f bearing=%+.0f°%@",
                     dec.choice.padding(toLength: 7, withPad: " ", startingAt: 0),
                     dec.probs.f, dec.probs.l, dec.probs.r, dec.probs.k,
                     ballDist, ballBearing * 180 / .pi, ballInView ? "" : " (视野外)"))
        switch dec.choice {
        case "FORWARD": vlmCmd = (0.3, 0); vlmActionUntil = now + 1.0
        case "LEFT", "RIGHT":
            var sign: Float = dec.choice == "LEFT" ? 1 : -1
            if !ballInView { sign = ballBearing > 0 ? 1 : -1 }   // 扫描纠偏：短边
            let dur = min(max(0.85 * abs(ballBearing) / 0.7, 0.08), 0.5)
            vlmCmd = (0, sign * 1.5)
            vlmActionUntil = now + Double(dur)
        case "KICK":
            vlmCmd = (0, 0); vlmActionUntil = now + 0.5
            // quackd kick：0.30m / ±35° 锥内 → 1.2 m/s 沿朝向（右腿 -0.05 rad 偏置
            // + N(0,6°) skew），纯水平冲量
            if ballDist <= 0.30 && abs(ballBearing) <= 35 * .pi / 180 {
                let skew = Float.random(in: -1...1) * 0.105 - 0.05
                let a = yaw + skew
                dd.pointee.qvel[ballV] = Double(1.2 * cos(a))
                dd.pointee.qvel[ballV + 1] = Double(1.2 * sin(a))
                print(String(format: "vlm: KICK 命中（dist=%.2f bearing=%.1f°）", ballDist, ballBearing * 180 / .pi))
            } else {
                print(String(format: "vlm: KICK 踢空（dist=%.2f bearing=%.1f°）", ballDist, ballBearing * 180 / .pi))
            }
        default: break
        }
    }
    var panelWindow: NSWindow?
    app.onDidFinishLaunching = {
        let w = control.makePanelWindow()
        panelWindow = w   // 持有引用，防释放
        w.makeKeyAndOrderFront(nil)
    }
    @MainActor func reframeSmooth() {
        applyFraming(fovDeg: Float(control.fovDeg), azDeg: Float(control.azDeg) + yawOffsetSmooth,
                     margin: Float(control.margin),
                     offsetX: Float(control.offsetX), offsetY: Float(control.offsetY))
    }
    /// 基座四元数（wxyz）→ 偏航角（度）
    @MainActor func yawDegOf(_ qw: Float, _ qx: Float, _ qy: Float, _ qz: Float) -> Float {
        atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) * 180 / .pi
    }
    // 行走 WASD 控制（幅值沿用 infer_policy.py：W/S=vx ±0.3、A/D=转向 ±1.5 rad/s、
    // Q/E=侧移 ±0.2、X=清零、空格=推倒；**按住才走，松开归零**）。文本框编辑中不抢键。
    app.onKey = { key in
        guard control.driveMode == .policy, !control.vlmDrive else { return false }
        if let fr = NSApp.keyWindow?.firstResponder, fr is NSTextView { return false }
        switch key {
        case "w", "🈁0": control.cmdVX = 0.3    // W / ↑
        case "s", "🈁1": control.cmdVX = -0.3   // S / ↓
        case "a", "🈁2": control.cmdYaw = 1.5   // A / ←  (左右 = 转向)
        case "d", "🈁3": control.cmdYaw = -1.5  // D / →
        case "q": control.cmdVY = 0.2           // Q/E 侧移
        case "e": control.cmdVY = -0.2
        case "x": control.cmdVX = 0; control.cmdVY = 0; control.cmdYaw = 0
                  control.headPitchDeg = 0; control.headYawDeg = 0
        case " ": control.pushRequest &+= 1   // 空格 = 随机推倒（看恢复策略起身）
        default: return false
        }
        return true
    }
    // 松开归零（按住才走）
    app.onKeyUp = { key in
        guard control.driveMode == .policy else { return }
        switch key {
        case "w", "s", "🈁0", "🈁1": control.cmdVX = 0
        case "a", "d", "🈁2", "🈁3": control.cmdYaw = 0
        case "q", "e": control.cmdVY = 0
        default: break
        }
    }
    app.onRenderQuilt = { cmd, _, _ in
        let nowF = CACurrentMediaTime()
        let frameDt = nowF - lastFrameT
        lastFrameT = nowF
        if control.cameraVersion != appliedCamVersion {
            appliedCamVersion = control.cameraVersion
            reframeSmooth()
        }
        var followXY: SIMD2<Float>?
        var followYawDeg: Float = 0
        if streamPort != nil {
            if let latest = udp.takeLatest() {
                if latest.qpos.count == nq {
                    for k in 0..<nq { d.pointee.qpos[k] = Double(latest.qpos[k]) }
                    d.pointee.time = Double(latest.simT)
                    mj_forward(m, d)
                    // 训练 env origins 是网格布点，env 0 可能离原点很远 → 跟随
                    followXY = SIMD2(latest.qpos[0], latest.qpos[1])
                    followYawDeg = yawDegOf(latest.qpos[3], latest.qpos[4],
                                            latest.qpos[5], latest.qpos[6])
                } else if !streamMismatchWarned {
                    streamMismatchWarned = true
                    print("--- stream: nq mismatch (got \(latest.qpos.count), viewer model nq=\(nq))，忽略后续包")
                }
            }
        } else {
            // 模式切换（自由物理 ⇄ 策略驾驶，原地复位）；换策略走热切换（下条）
            if control.driveModeVersion != appliedDriveVersion
                || control.duckVersion != appliedDuckVersion {
                appliedDriveVersion = control.driveModeVersion
                appliedDuckVersion = control.duckVersion
                appliedPolicyVersion = control.policyVersion
                if control.driveMode == .policy {
                    setupDucks(control.duckCount)
                    if policies.isEmpty {
                        print("--- policy 加载失败，回退自由物理")
                        control.driveMode = .free
                        appliedDriveVersion = control.driveModeVersion
                    }
                } else {
                    bamEdit.restore(d: d)   // 自由物理回 XML 原语义（若 BAM 改写曾应用）
                    m.pointee.opt.timestep = origTimestep
                    floorFollow = false
                    if mm.nkey > 0 { mj_resetDataKeyframe(m, d, 0) }
                    else { mj_resetData(m, d); mj_forward(m, d) }   // arena 无 keyframe
                }
            } else if control.driveMode == .policy && control.policyVersion != appliedPolicyVersion {
                // 策略热切换（真机 runtime 语义：61D obs 契约共享，物理状态不动）：
                // 只重建 PolicyRunner；bam.reset() 把 qTarget 锚到当前关节位，防猛拽
                appliedPolicyVersion = control.policyVersion
                policies = datas.compactMap { PolicyRunner(m: m, d: $0, onnxPath: control.policyPath,
                                                           useBam: control.actuatorBAM) }
                for pol in policies { pol.bam?.reset() }
                if policies.isEmpty {
                    print("--- policy 热切换加载失败，回退自由物理")
                    control.driveMode = .free
                    appliedDriveVersion = control.driveModeVersion
                } else {
                    print("--- policy hot-swap（不复位）: \(control.policyPath.split(separator: "/").last!)")
                }
            }
            if control.resetRequest != appliedReset {
                appliedReset = control.resetRequest
                if control.driveMode == .policy { setupDucks(policies.count, respawnBall: true) }
                else if mm.nkey > 0 { mj_resetDataKeyframe(m, d, 0) }
                else { mj_resetData(m, d); mj_forward(m, d) }
            }
            if control.pushRequest != appliedPush {
                appliedPush = control.pushRequest
                if control.driveMode == .policy {
                    // infer_policy P 键：trunk xy 随机方向 1.0 m/s 冲击（每只各随一个）
                    for dd in datas {
                        let a = Float.random(in: 0 ..< 2 * .pi)
                        dd.pointee.qvel[0] = Double(cos(a))
                        dd.pointee.qvel[1] = Double(sin(a))
                    }
                }
            }
            if !control.physicsPaused {
                if control.driveMode == .policy, !policies.isEmpty {
                    // VLM 驾驶：sidecar 决策覆盖键盘指令（到期清零）；开关切换时同步生命周期
                    if control.vlmDrive && arenaVLM {
                        if !vlm.ready && !vlm.inFlight { vlm.start() }
                        let nowV = CACurrentMediaTime()
                        let active = nowV < vlmActionUntil
                        control.cmdVX = Double(active ? vlmCmd.vx : 0)
                        control.cmdVY = 0
                        control.cmdYaw = Double(active ? vlmCmd.vyaw : 0)
                        // ---- 独立头部注视控制器（VLM 驾驶时接管脖子）----
                        // VLM 输入走躯干锁定虚拟相机（headHomeRel），脖子动作不影响决策
                        // 分布。球在近旁/可转及范围 → 注视（yaw 准，pitch ±25° 内粗跟）；
                        // 否则缓慢扫描（ trunk 系 ±40° @0.25Hz 正弦）。
                        if !datas.isEmpty {
                            let dd0 = datas[0]
                            let ci0 = Int(headCamId)
                            let hx = Float(dd0.pointee.cam_xpos[ci0 * 3])
                            let hy = Float(dd0.pointee.cam_xpos[ci0 * 3 + 1])
                            let hz = Float(dd0.pointee.cam_xpos[ci0 * 3 + 2])
                            let bx = Float(dd0.pointee.qpos[ballQ]), by = Float(dd0.pointee.qpos[ballQ + 1])
                            let qw0 = Float(dd0.pointee.qpos[3]), qx0 = Float(dd0.pointee.qpos[4]),
                                qy0 = Float(dd0.pointee.qpos[5]), qz0 = Float(dd0.pointee.qpos[6])
                            let yaw0 = atan2f(2 * (qw0 * qz0 + qx0 * qy0), 1 - 2 * (qy0 * qy0 + qz0 * qz0))
                            var brg = atan2f(by - hy, bx - hx) - yaw0
                            while brg > .pi { brg -= 2 * .pi }
                            while brg < -.pi { brg += 2 * .pi }
                            let dist0 = hypotf(bx - hx, by - hy)
                            let track = dist0 < 1.3 && abs(brg) < 1.2   // ~69° 内可转及
                            var yawT: Float = 0, pitchT: Float = 0
                            if track {
                                yawT = min(max(brg * 180 / .pi, -55), 55)
                                pitchT = min(max(atan2f(hz - 0.05, dist0) * 180 / .pi, 0), 25)
                            } else {
                                yawT = 40 * sinf(Float(dd0.pointee.time) * 2 * .pi * 0.25)
                            }
                            let kg = Float(1 - exp(-frameDt / 0.25))
                            control.headYawDeg += Double((yawT - Float(control.headYawDeg)) * kg)
                            control.headPitchDeg += Double((pitchT - Float(control.headPitchDeg)) * kg)
                        }
                    } else if vlm.ready {
                        vlm.stop()
                    }
                    // 50Hz 策略拍（写指令 → tick），随后 4×200Hz 物理子步逐只交错
                    //（BAM 摩擦参数写在共享 mjModel 上，update→step 必须按鸭紧邻）
                    var ticked: [PolicyRunner] = []
                    for pol in policies {
                        pol.command.vx = Float(control.cmdVX)
                        pol.command.vy = Float(control.cmdVY)
                        pol.command.vyaw = Float(control.cmdYaw)
                        // 头部命令（quackd look 语义：head_pitch 正=低头、head_yaw 正=左）
                        pol.command.head = SIMD4(0, Float(control.headPitchDeg) * .pi / 180,
                                                 Float(control.headYawDeg) * .pi / 180, 0)
                        if pol.tick(wallDt: frameDt) { ticked.append(pol) }
                    }
                    if !ticked.isEmpty {
                        for _ in 0..<4 { for pol in ticked { pol.stepPhysics() } }
                    }
                    // 相机跟鸭群质心（广播指令下近似同步移动）
                    var mx: Float = 0, my: Float = 0
                    for dd in datas { mx += Float(dd.pointee.qpos[0]); my += Float(dd.pointee.qpos[1]) }
                    let inv = 1 / Float(datas.count)
                    followXY = SIMD2(mx * inv, my * inv)
                    followYawDeg = yawDegOf(Float(datas[0].pointee.qpos[3]), Float(datas[0].pointee.qpos[4]),
                                            Float(datas[0].pointee.qpos[5]), Float(datas[0].pointee.qpos[6]))
                    // 球贴墙静止 0.8s → 重布（否则鸭子一直怼墙追球）
                    if ballQ >= 0 {
                        let dd = datas[0]
                        let bx = dd.pointee.qpos[ballQ], by = dd.pointee.qpos[ballQ + 1]
                        let spd = hypot(dd.pointee.qvel[ballV], dd.pointee.qvel[ballV + 1])
                        if (abs(bx) > 0.88 || abs(by) > 0.88) && spd < 0.02 {
                            ballWallRestT += frameDt
                            if ballWallRestT > 0.8 {
                                ballWallRestT = 0
                                placeBall()
                            }
                        } else { ballWallRestT = 0 }
                    }
                } else {
                    for _ in 0..<control.stepsPerFrame { mj_step(m, d) }
                }
            }
        }
        // 相机跟随：平滑逼近 base xy（τ=0.25s）；朝向跟随（τ=1.5s 慢速，
        // 吃掉步态自身的偏航摆动；az 滑杆语义变为偏置）——关掉则回正到滑杆角度
        if let fxy = followXY {
            if control.followSmooth {
                let k = Float(1 - exp(-frameDt / 0.25))
                center += (SIMD3(fxy.x, fxy.y, center0.z) - center) * k
            } else {
                center = SIMD3(fxy.x, fxy.y, center0.z)
            }
            let kYaw = Float(1 - exp(-frameDt / 1.5))
            if control.followHeading {
                var dy = followYawDeg - yawOffsetSmooth
                dy = (dy + 180).truncatingRemainder(dividingBy: 360)
                if dy < -180 { dy += 360 }
                yawOffsetSmooth += dy * kYaw
            } else {
                yawOffsetSmooth -= yawOffsetSmooth * kYaw
            }
            reframeSmooth()
        }
        var items: [DrawItem] = []
        if streamPort == nil && control.driveMode == .policy {
            for i in 0..<datas.count {
                mjv_updateScene(m, datas[i], &vopt, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scns[i])
                items += collectDrawItems(from: &scns[i], includePlane: i == 0).items
            }
        } else {
            mjv_updateScene(m, d, &vopt, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scn)
            items = collectDrawItems(from: &scn).items
        }
        // VLM head-cam：把鸭子 0 的头相机渲进 headTex（GPU 常驻，仅查询时回读
        // 上一帧内容——60fps 下 16ms 延迟对 1-2Hz 决策无感）
        if streamPort == nil && control.driveMode == .policy && control.vlmDrive && arenaVLM
            && !datas.isEmpty && vlm.ready {
            mjv_updateScene(m, datas[0], &voptHead, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scnHead)
            // VLM 输入 = 躯干锁定虚拟相机（脖子动作不出分布；见 headCamViewProjTrunkLocked）
            let (hvp, heye, hfwd) = headHomeRel.map {
                headCamViewProjTrunkLocked(d: datas[0], camId: headCamId, trunkId: trunkBodyId, rel: $0)
            } ?? headCamViewProj(d: datas[0], camId: headCamId)
            let hItems = collectDrawItems(from: &scnHead).items
            renderer.encodeHeadCamera(cmd: cmd, items: hItems, viewProj: hvp,
                                      camPos: heye, camFwd: hfwd)
            let nowV = CACurrentMediaTime()
            vlm.watchdog()
            if !vlm.inFlight && nowV >= vlmActionUntil {
                let pix = renderer.readHeadPixels()
                var rgb = [UInt8](repeating: 0, count: 256 * 256 * 3)
                for px in 0..<(256 * 256) {
                    rgb[px * 3] = pix[px * 4]
                    rgb[px * 3 + 1] = pix[px * 4 + 1]
                    rgb[px * 3 + 2] = pix[px * 4 + 2]
                }
                vlm.send(rgb: rgb)
                // 面板预览（VLM 所见）
                if let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 256,
                                              pixelsHigh: 256, bitsPerSample: 8,
                                              samplesPerPixel: 4, hasAlpha: true,
                                              isPlanar: false, colorSpaceName: .deviceRGB,
                                              bytesPerRow: 256 * 4, bitsPerPixel: 32) {
                    pix.withUnsafeBytes { memcpy(rep.bitmapData!, $0.baseAddress!, pix.count) }
                    let img = NSImage(size: NSSize(width: 256, height: 256))
                    img.addRepresentation(rep)
                    control.vlmFrame = img
                }
            }
        }
        renderer.encodeScene(cmd: cmd, items: items)
        ctlFrames += 1
        let nowT = CACurrentMediaTime()
        if nowT - ctlLastReport >= 0.25 {
            let pkts = udp.packetCount
            control.fps = Double(ctlFrames) / (nowT - ctlLastReport)
            control.qpos = qposNorm()
            control.simT = d.pointee.time
            control.draws = renderer.drawCalls
            control.tris = renderer.triangles
            if streamPort != nil {
                let pps = Double(pkts - lastPacketCount) / (nowT - ctlLastReport)
                control.distInfo = String(format: "stream :%d · %.1f pkt/s · quilt %dx%d %@",
                                          streamPort!, pps,
                                          quilt.quiltTexture.width, quilt.quiltTexture.height, layout)
                lastPacketCount = pkts
            } else {
                control.distInfo = String(format: "quilt %dx%d %@ · dist %.3f sweep %.3f",
                                          quilt.quiltTexture.width, quilt.quiltTexture.height,
                                          layout, dist, camera.sweep)
            }
            ctlFrames = 0
            ctlLastReport = nowT
        }
    }
    app.onStatusLine = {
        String(format: "|qpos| %.4f sim t=%.2fs draws=%d tris=%d",
               qposNorm(), d.pointee.time, renderer.drawCalls, renderer.triangles)
    }
    print("=== app mode: LKG 设备窗 60Hz + 主屏预览窗 + 控制面板（c 测试图 / b 裸 quilt 上屏 / g 网格 / p 暂停 / Cmd-Q 退出）")
    app.run()
}

mjv_freeScene(&scn)
mj_deleteData(d)
mj_deleteModel(m)
