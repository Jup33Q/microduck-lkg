import Foundation
import MuJoCo

// P0 probe：libmujoco 接入验证（headless，无渲染）
// 用法: swift run mujoco-lkg-probe [scene.xml] [--steps N]
// 默认场景 = microduck scene.xml（与 infer_policy.py 的 MICRODUCK_XML 同源）

let defaultXML = "/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"

var xmlPath = defaultXML
var steps = 1000
var args = CommandLine.arguments.dropFirst()
while let a = args.popFirst() {
    if a == "--steps", let n = args.popFirst(), let v = Int(n) { steps = v }
    else if !a.hasPrefix("--") { xmlPath = a }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(1)
}

// ---- 载入模型 ------------------------------------------------------------
var errBuf = [CChar](repeating: 0, count: 1000)
guard let m = mj_loadXML(xmlPath, nil, &errBuf, 1000) else {
    fail("mj_loadXML failed: \(String(cString: errBuf))")
}
let d = mj_makeData(m)
guard let d else { fail("mj_makeData failed") }

let mm = m.pointee
print("=== model: \(xmlPath)")
print("nq=\(mm.nq) nv=\(mm.nv) nu=\(mm.nu) nbody=\(mm.nbody) njnt=\(mm.njnt) ngeom=\(mm.ngeom) nmesh=\(mm.nmesh) nmat=\(mm.nmat) ntex=\(mm.ntex) nkey=\(mm.nkey)")
print("timestep=\(mm.opt.timestep) solver=\(mm.opt.solver) iterations=\(mm.opt.iterations)")

func cname(_ base: UnsafeMutablePointer<CChar>?, _ adr: Int32) -> String {
    guard let base, adr >= 0 else { return "<none>" }
    return String(cString: base.advanced(by: Int(adr)))
}

// ---- 名称清单（与 Python 侧 mj_name2id 对照用） ---------------------------
print("--- joints (\(mm.njnt))")
for j in 0..<Int(mm.njnt) {
    let name = cname(mm.names, mm.name_jntadr[j])
    let pyid = mj_name2id(m, Int32(mjOBJ_JOINT.rawValue), name)
    print("  jnt[\(j)] \(name)  type=\(mm.jnt_type[j]) qposadr=\(mm.jnt_qposadr[j]) mj_name2id=\(pyid)")
}
print("--- bodies (\(mm.nbody))")
for b in 0..<Int(mm.nbody) {
    print("  body[\(b)] \(cname(mm.names, mm.name_bodyadr[b]))")
}
print("--- model geoms (\(mm.ngeom))")
for g in 0..<Int(mm.ngeom) {
    print("  geom[\(g)] \(cname(mm.names, mm.name_geomadr[g])) type=\(mm.geom_type[g]) bodyid=\(mm.geom_bodyid[g]) meshid=\(mm.geom_dataid[g]) matid=\(mm.geom_matid[g])")
}
print("--- meshes (\(mm.nmesh))")
for i in 0..<Int(mm.nmesh) {
    let vn = mm.mesh_vertnum != nil ? Int(mm.mesh_vertnum[i]) : -1
    let fn = mm.mesh_facenum != nil ? Int(mm.mesh_facenum[i]) : -1
    print("  mesh[\(i)] \(cname(mm.names, mm.name_meshadr[i])) vertnum=\(vn) facenum=\(fn)")
}

// ---- keyframe INIT → mj_forward → 记录初始 qpos ---------------------------
if mm.nkey > 0 { mj_resetDataKeyframe(m, d, 0) }
mj_forward(m, d)

let nq = Int(mm.nq)
var qpos0 = [Double](repeating: 0, count: nq)
for i in 0..<nq { qpos0[i] = d.pointee.qpos[i] }
print("--- qpos0: \(qpos0.map { String(format: "%.4f", $0) }.joined(separator: " "))")

// ---- mj_step × N，计时 ----------------------------------------------------
var times = [Double]()
times.reserveCapacity(steps)
for _ in 0..<steps {
    let t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    mj_step(m, d)
    let t1 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    times.append(Double(t1 - t0) / 1e6)  // ms
}
times.sort()
let sum = times.reduce(0, +)
print("--- mj_step x\(steps): mean=\(String(format: "%.4f", sum / Double(steps)))ms min=\(String(format: "%.4f", times.first!))ms p50=\(String(format: "%.4f", times[steps / 2]))ms max=\(String(format: "%.4f", times.last!))ms")
print("    sim time advanced: \(String(format: "%.3f", d.pointee.time))s (\(steps) x \(mm.opt.timestep)s)")

var dq = 0.0
for i in 0..<nq { let diff = d.pointee.qpos[i] - qpos0[i]; dq += diff * diff }
print("--- |qpos_after - qpos0| = \(String(format: "%.6f", dq.squareRoot()))")
print("    qpos_after: \((0..<nq).map { String(format: "%.4f", d.pointee.qpos[$0]) }.joined(separator: " "))")

// ---- mjvScene 走查（P1 光栅器的数据源确认） -------------------------------
var scn = mjvScene()
mjv_defaultScene(&scn)
var vopt = mjvOption()
mjv_defaultOption(&vopt)
var vcam = mjvCamera()
mjv_defaultFreeCamera(m, &vcam)
mjv_makeScene(m, &scn, 20000)
mj_forward(m, d)
mjv_updateScene(m, d, &vopt, nil, &vcam, Int32(mjCAT_ALL.rawValue), &scn)
print("=== mjvScene: ngeom=\(scn.ngeom) maxgeom=20000")

func gi(_ v: mjtGeom) -> Int32 { Int32(v.rawValue) }
let geomTypeName: [Int32: String] = [
    gi(mjGEOM_PLANE): "plane", gi(mjGEOM_HFIELD): "hfield",
    gi(mjGEOM_SPHERE): "sphere", gi(mjGEOM_CAPSULE): "capsule",
    gi(mjGEOM_ELLIPSOID): "ellipsoid", gi(mjGEOM_CYLINDER): "cylinder",
    gi(mjGEOM_BOX): "box", gi(mjGEOM_MESH): "mesh",
]
var typeCount = [String: Int]()
var meshSeen = Set<Int32>()
for i in 0..<Int(scn.ngeom) {
    let g = scn.geoms[i]
    let tname = geomTypeName[g.type] ?? "type\(g.type)"
    typeCount[tname, default: 0] += 1
    if g.type == gi(mjGEOM_MESH) { meshSeen.insert(g.dataid) }
    if i < 12 || g.type == gi(mjGEOM_MESH) {
        print(String(format: "  scn.geom[%d] %@ objtype=%d objid=%d meshid=%d matid=%d pos=(%.3f %.3f %.3f) size=(%.3f %.3f %.3f)",
                     i, tname, g.objtype, g.objid, g.dataid, g.matid,
                     g.pos.0, g.pos.1, g.pos.2, g.size.0, g.size.1, g.size.2))
    }
}
print("--- scn geom type histogram: \(typeCount.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: " "))")
print("--- distinct mesh ids referenced: \(meshSeen.sorted())")
mjv_freeScene(&scn)

mj_deleteData(d)
mj_deleteModel(m)
print("=== P0 probe OK")
