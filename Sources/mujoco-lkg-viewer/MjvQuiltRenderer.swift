import Foundation
import Metal
import simd
import MuJoCo
import StreamQuilt

// P1: mjvScene → Metal 光栅器。
// mesh 上传（mjModel.mesh_* → MTLBuffer）、逐 tile 离轴投影 quilt 渲染。
// quilt 布局/离轴相机数学全部复用 StreamQuilt（QuiltSpec/QuiltCamera/QuiltRenderer）。

// Swift 侧顶点布局，须与 MSL 端 `struct Vtx` 一致（32 字节）。
struct MjvVertex {
    var px: Float; var py: Float; var pz: Float
    var nx: Float; var ny: Float; var nz: Float
    var u: Float; var v: Float
}

struct ViewUniforms {
    var viewProj: float4x4
    var camPos: SIMD4<Float>   // 相机帧（=worldFromMujoco 之后的空间）下的相机位置
    var params: SIMD4<Float>   // x: 光照模式（0=quilt 线性+ACES，1=quackd 头灯 gamma 朴素）
    var camFwd: SIMD4<Float>   // 相机前向（quackd 头灯方向；仅 head-cam 模式用）
}

struct InstanceUniforms {
    var model: float4x4        // worldFromMujoco ∘ geomPose（mesh frame 已在顶点里烘焙）
    var color: SIMD4<Float>
    var misc: SIMD4<Float>     // specular, shininess, emission, unused
}

struct GPUMesh {
    let smoothVertices: MTLBuffer
    let smoothIndices: MTLBuffer
    let smoothIndexCount: Int
    let flatVertices: MTLBuffer
    let flatCount: Int
}

enum DrawKind {
    case meshSmooth(GPUMesh)
    case meshFlat(GPUMesh)
    case plane
    case sphere
    case box
}

struct DrawItem {
    var kind: DrawKind
    var model: float4x4        // MuJoCo 世界系下的 geom 位姿
    var rgba: SIMD4<Float>
    var misc: SIMD4<Float>
}

// ---- C 数组/tuple 取值辅助 ---------------------------------------------------

func floatsOf<T>(_ t: T, _ n: Int) -> [Float] {
    withUnsafeBytes(of: t) { Array($0.bindMemory(to: Float.self).prefix(n)) }
}

func mat4(_ r: float3x3, _ t: SIMD3<Float>) -> float4x4 {
    float4x4(columns: (
        SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0), SIMD4(r.columns.2, 0),
        SIMD4(t, 1)
    ))
}

/// 绕 z 轴旋转（Matrix4x4 只有 rotationX/Y，这里补 rotZ；非 quilt 常量）。
func rotationZ(_ a: Float) -> float4x4 {
    float4x4(columns: (
        SIMD4(cos(a), sin(a), 0, 0), SIMD4(-sin(a), cos(a), 0, 0),
        SIMD4(0, 0, 1, 0), SIMD4(0, 0, 0, 1)
    ))
}

// ---- mesh 上传 ---------------------------------------------------------------

/// mjModel.mesh_* → 每 mesh 双 MTLBuffer（smooth=索引化+顶点法线 / flat=展开+面法线，
/// 按 scene geom `dataid&1` 选择，`dataid/2`=meshid）。
/// 注意：mesh_vert/mesh_normal 在 mjC 编译期已烘焙 mesh_scale/mesh_pos/mesh_quat
///（头文件注释 "applied to asset vertices"），即顶点已在 geom frame——
/// **不要再乘 mesh frame**（曾二次变换导致部件散架；--diag 验证 mjvScene pose
/// 与 geom_xpos/xmat 完全一致）。绘制时只需乘 geom 世界位姿。
func uploadMeshes(m: mjModel, device: MTLDevice) -> [GPUMesh] {
    var out: [GPUMesh] = []
    out.reserveCapacity(Int(m.nmesh))
    for i in 0..<Int(m.nmesh) {
        let vadr = Int(m.mesh_vertadr[i]), vnum = Int(m.mesh_vertnum[i])
        let nadr = m.mesh_normaladr != nil ? Int(m.mesh_normaladr[i]) : -1
        let nnum = m.mesh_normalnum != nil ? Int(m.mesh_normalnum[i]) : 0
        let tadr = m.mesh_texcoordadr != nil ? Int(m.mesh_texcoordadr[i]) : -1
        let fadr = Int(m.mesh_faceadr[i]), fnum = Int(m.mesh_facenum[i])

        // geom frame 顶点（编译期已烘焙，直接读）
        var pos = [SIMD3<Float>](repeating: .zero, count: vnum)
        for v in 0..<vnum {
            pos[v] = SIMD3(m.mesh_vert[(vadr + v) * 3],
                           m.mesh_vert[(vadr + v) * 3 + 1],
                           m.mesh_vert[(vadr + v) * 3 + 2])
        }

        // smooth 法线：优先 mesh_normal；缺省/数量不符时按面累积重算
        var nrm = [SIMD3<Float>](repeating: .zero, count: vnum)
        if nnum == vnum && nadr >= 0 && m.mesh_normal != nil {
            for v in 0..<vnum {
                nrm[v] = SIMD3(m.mesh_normal[(nadr + v) * 3],
                               m.mesh_normal[(nadr + v) * 3 + 1],
                               m.mesh_normal[(nadr + v) * 3 + 2])
            }
        } else {
            for f in 0..<fnum {
                let a = Int(m.mesh_face[(fadr + f) * 3])
                let b = Int(m.mesh_face[(fadr + f) * 3 + 1])
                let c = Int(m.mesh_face[(fadr + f) * 3 + 2])
                let fn = cross(pos[b] - pos[a], pos[c] - pos[a])
                nrm[a] += fn; nrm[b] += fn; nrm[c] += fn
            }
            for v in 0..<vnum {
                let l = length(nrm[v])
                nrm[v] = l > 1e-12 ? nrm[v] / l : SIMD3(0, 0, 1)
            }
        }

        var smoothVerts = [MjvVertex](repeating: MjvVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 1, u: 0, v: 0), count: vnum)
        for v in 0..<vnum {
            var uv = SIMD2<Float>.zero
            if tadr >= 0 && v < (m.mesh_texcoordnum != nil ? Int(m.mesh_texcoordnum[i]) : 0) {
                uv = SIMD2(m.mesh_texcoord[(tadr + v) * 2], m.mesh_texcoord[(tadr + v) * 2 + 1])
            }
            smoothVerts[v] = MjvVertex(px: pos[v].x, py: pos[v].y, pz: pos[v].z,
                                       nx: nrm[v].x, ny: nrm[v].y, nz: nrm[v].z,
                                       u: uv.x, v: uv.y)
        }
        var indices = [UInt32]()
        indices.reserveCapacity(fnum * 3)
        for f in 0..<(fnum * 3) { indices.append(UInt32(bitPattern: m.mesh_face[fadr * 3 + f])) }

        // flat 变体：展开为每面 3 顶点，面法线
        var flatVerts = [MjvVertex]()
        flatVerts.reserveCapacity(fnum * 3)
        for f in 0..<fnum {
            let a = Int(m.mesh_face[(fadr + f) * 3])
            let b = Int(m.mesh_face[(fadr + f) * 3 + 1])
            let c = Int(m.mesh_face[(fadr + f) * 3 + 2])
            let fn = normalize(cross(pos[b] - pos[a], pos[c] - pos[a]))
            for idx in [a, b, c] {
                flatVerts.append(MjvVertex(px: pos[idx].x, py: pos[idx].y, pz: pos[idx].z,
                                           nx: fn.x, ny: fn.y, nz: fn.z,
                                           u: smoothVerts[idx].u, v: smoothVerts[idx].v))
            }
        }

        guard let sv = device.makeBuffer(bytes: smoothVerts, length: vnum * 32),
              let ib = device.makeBuffer(bytes: indices, length: indices.count * 4),
              let fb = device.makeBuffer(bytes: flatVerts, length: flatVerts.count * 32)
        else { fatalError("mesh buffer alloc failed for mesh \(i)") }
        out.append(GPUMesh(smoothVertices: sv, smoothIndices: ib, smoothIndexCount: indices.count,
                           flatVertices: fb, flatCount: flatVerts.count))
    }
    return out
}

// ---- 光栅器 -------------------------------------------------------------------

final class MjvQuiltRenderer {
    let quilt: QuiltRenderer
    let device: MTLDevice
    var camera: QuiltCamera
    var nearPlane: Float = 0.01
    var farPlane: Float = 100
    /// MuJoCo 世界 → 相机帧（z-up → y-up、主体居中、朝向摆正）。
    var worldFromMujoco = matrix_identity_float4x4

    private let pso: MTLRenderPipelineState
    private let psoRgba8: MTLRenderPipelineState   // head-cam：256×256 rgba8（无 tonemap）
    private let depthState: MTLDepthStencilState
    private var depthTexture: MTLTexture?
    private var headDepthTexture: MTLTexture?
    private let planeBuffer: MTLBuffer
    private let planeVertCount = 6
    // 图元：单位球（r=1，索引化 smooth）/ 单位立方体（半边长 1，面法线展开）
    private let sphereVerts: MTLBuffer
    private let sphereIndices: MTLBuffer
    private let sphereIndexCount: Int
    private let boxVerts: MTLBuffer
    private let boxVertCount: Int
    /// head-cam 渲染目标（256×256 rgba8Unorm shared，可回读）
    let headTex: MTLTexture

    private(set) var drawCalls = 0
    private(set) var triangles = 0

    init(quilt: QuiltRenderer, camera: QuiltCamera) throws {
        self.quilt = quilt
        self.device = quilt.device
        self.camera = camera

        let lib = try quilt.device.makeLibrary(source: Self.shaderSource, options: nil)
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "mjvVS")
        d.fragmentFunction = lib.makeFunction(name: "mjvFS")
        d.colorAttachments[0].pixelFormat = .rgba16Float
        d.depthAttachmentPixelFormat = .depth32Float
        // microduck STL 网格朝向未验证，关掉背面剔除（代价可忽略，视觉正确性优先）。
        pso = try quilt.device.makeRenderPipelineState(descriptor: d)

        let d8 = MTLRenderPipelineDescriptor()
        d8.vertexFunction = lib.makeFunction(name: "mjvVS")
        d8.fragmentFunction = lib.makeFunction(name: "mjvFS")
        d8.colorAttachments[0].pixelFormat = .rgba8Unorm
        d8.depthAttachmentPixelFormat = .depth32Float
        psoRgba8 = try quilt.device.makeRenderPipelineState(descriptor: d8)

        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less
        ds.isDepthWriteEnabled = true
        depthState = quilt.device.makeDepthStencilState(descriptor: ds)!

        // 单位地面四边形（geom frame 内 z=0、法线 +z），绘制时按 geom size 缩放
        let quad: [MjvVertex] = [
            MjvVertex(px: -1, py: -1, pz: 0, nx: 0, ny: 0, nz: 1, u: 0, v: 0),
            MjvVertex(px:  1, py: -1, pz: 0, nx: 0, ny: 0, nz: 1, u: 1, v: 0),
            MjvVertex(px:  1, py:  1, pz: 0, nx: 0, ny: 0, nz: 1, u: 1, v: 1),
            MjvVertex(px: -1, py: -1, pz: 0, nx: 0, ny: 0, nz: 1, u: 0, v: 0),
            MjvVertex(px:  1, py:  1, pz: 0, nx: 0, ny: 0, nz: 1, u: 1, v: 1),
            MjvVertex(px: -1, py:  1, pz: 0, nx: 0, ny: 0, nz: 1, u: 0, v: 1),
        ]
        planeBuffer = quilt.device.makeBuffer(bytes: quad, length: quad.count * 32)!

        // 单位球（r=1）：24 经 × 12 纬，法线 = 顶点方向
        var sv: [MjvVertex] = []
        var si: [UInt32] = []
        let NLon = 24, NLat = 12
        for lat in 0...NLat {
            let th = Float(lat) / Float(NLat) * Float.pi
            for lon in 0...NLon {
                let ph = Float(lon) / Float(NLon) * 2 * Float.pi
                let n = SIMD3<Float>(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
                sv.append(MjvVertex(px: n.x, py: n.y, pz: n.z, nx: n.x, ny: n.y, nz: n.z, u: 0, v: 0))
            }
        }
        for lat in 0..<NLat {
            for lon in 0..<NLon {
                let a = UInt32(lat * (NLon + 1) + lon), b = a + UInt32(NLon + 1)
                si += [a, b, a + 1, a + 1, b, b + 1]
            }
        }
        sphereVerts = quilt.device.makeBuffer(bytes: sv, length: sv.count * 32)!
        sphereIndices = quilt.device.makeBuffer(bytes: si, length: si.count * 4)!
        sphereIndexCount = si.count

        // 单位立方体（半边长 1）：6 面 × 2 三角形，面法线
        var bv: [MjvVertex] = []
        func quad4(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ d4: SIMD3<Float>) {
            let n = normalize(cross(b - a, c - a))
            for p in [a, b, c, a, c, d4] {
                bv.append(MjvVertex(px: p.x, py: p.y, pz: p.z, nx: n.x, ny: n.y, nz: n.z, u: 0, v: 0))
            }
        }
        quad4(SIMD3(-1, -1, 1), SIMD3(1, -1, 1), SIMD3(1, 1, 1), SIMD3(-1, 1, 1))       // +z
        quad4(SIMD3(1, -1, -1), SIMD3(-1, -1, -1), SIMD3(-1, 1, -1), SIMD3(1, 1, -1))   // -z
        quad4(SIMD3(1, -1, 1), SIMD3(1, -1, -1), SIMD3(1, 1, -1), SIMD3(1, 1, 1))       // +x
        quad4(SIMD3(-1, -1, -1), SIMD3(-1, -1, 1), SIMD3(-1, 1, 1), SIMD3(-1, 1, -1))   // -x
        quad4(SIMD3(-1, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, -1), SIMD3(-1, 1, -1))       // +y
        quad4(SIMD3(-1, -1, -1), SIMD3(1, -1, -1), SIMD3(1, -1, 1), SIMD3(-1, -1, 1))   // -y
        boxVerts = quilt.device.makeBuffer(bytes: bv, length: bv.count * 32)!
        boxVertCount = bv.count

        let hd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 256, height: 256, mipmapped: false)
        hd.usage = [.renderTarget]
        hd.storageMode = .shared
        headTex = quilt.device.makeTexture(descriptor: hd)!
    }

    private func makePassDescriptor() -> MTLRenderPassDescriptor {
        let w = quilt.quiltTexture.width, h = quilt.quiltTexture.height
        if depthTexture == nil || depthTexture!.width != w || depthTexture!.height != h {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: w, height: h, mipmapped: false)
            d.usage = [.renderTarget]
            d.storageMode = .private
            depthTexture = device.makeTexture(descriptor: d)
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = quilt.quiltTexture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.012, green: 0.018, blue: 0.035, alpha: 1)
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        return pass
    }

    /// 视角 view 的 viewProj（相机帧）。view 0 = 左下 tile，由 QuiltRenderer.tileRect 保证。
    func viewProj(view: Int) -> float4x4 {
        let n = quilt.spec.viewCount
        let off = camera.viewOffset(index: view, viewCount: n)
        return camera.projectionMatrix(tileAspect: quilt.spec.tileAspect,
                                       near: nearPlane, far: farPlane, viewOffset: off)
            * camera.viewMatrix(viewOffset: off)
    }

    /// 焦点平面像素级验证：世界原点（=主体中心，焦点平面 z=0）在所有视角下
    /// 必须投到同一 tile 像素。返回跨视角最大像素偏差。
    func focusPlanePixelError() -> Float {
        var ref: SIMD2<Float>?
        var maxErr: Float = 0
        let tw = Float(quilt.quiltTexture.width) / Float(quilt.spec.columns)
        let th = Float(quilt.quiltTexture.height) / Float(quilt.spec.rows)
        for v in 0..<quilt.spec.viewCount {
            let clip = viewProj(view: v) * SIMD4<Float>(0, 0, 0, 1)
            let ndc = SIMD2(clip.x, clip.y) / clip.w
            if let r = ref {
                let d = abs(ndc - r) * 0.5 * SIMD2(tw, th)
                maxErr = max(maxErr, max(d.x, d.y))
            } else { ref = ndc }
        }
        return maxErr
    }

    func encodeScene(cmd: MTLCommandBuffer, items: [DrawItem]) {
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: makePassDescriptor()) else { return }
        enc.setRenderPipelineState(pso)
        enc.setDepthStencilState(depthState)
        drawCalls = 0; triangles = 0
        let n = quilt.spec.viewCount
        for view in 0..<n {
            let vp = quilt.tileRect(index: view)
            enc.setViewport(vp)
            enc.setScissorRect(MTLScissorRect(x: Int(vp.originX), y: Int(vp.originY),
                                              width: Int(vp.width), height: Int(vp.height)))
            let off = camera.viewOffset(index: view, viewCount: n)
            var vu = ViewUniforms(viewProj: viewProj(view: view),
                                  camPos: SIMD4(off, 0, camera.distance, 1),
                                  params: SIMD4(0, 0, 0, 0), camFwd: SIMD4(0, 0, 1, 0))
            enc.setVertexBytes(&vu, length: MemoryLayout<ViewUniforms>.stride, index: 1)
            enc.setFragmentBytes(&vu, length: MemoryLayout<ViewUniforms>.stride, index: 1)
            for item in items {
                var iu = InstanceUniforms(model: worldFromMujoco * item.model,
                                          color: item.rgba, misc: item.misc)
                enc.setVertexBytes(&iu, length: MemoryLayout<InstanceUniforms>.stride, index: 2)
                enc.setFragmentBytes(&iu, length: MemoryLayout<InstanceUniforms>.stride, index: 2)
                switch item.kind {
                case .meshSmooth(let g):
                    enc.setVertexBuffer(g.smoothVertices, offset: 0, index: 0)
                    enc.drawIndexedPrimitives(type: .triangle, indexCount: g.smoothIndexCount,
                                              indexType: .uint32, indexBuffer: g.smoothIndices,
                                              indexBufferOffset: 0)
                    drawCalls += 1; triangles += g.smoothIndexCount / 3
                case .meshFlat(let g):
                    enc.setVertexBuffer(g.flatVertices, offset: 0, index: 0)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: g.flatCount)
                    drawCalls += 1; triangles += g.flatCount / 3
                case .plane:
                    enc.setVertexBuffer(planeBuffer, offset: 0, index: 0)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: planeVertCount)
                    drawCalls += 1; triangles += 2
                case .sphere:
                    enc.setVertexBuffer(sphereVerts, offset: 0, index: 0)
                    enc.drawIndexedPrimitives(type: .triangle, indexCount: sphereIndexCount,
                                              indexType: .uint32, indexBuffer: sphereIndices,
                                              indexBufferOffset: 0)
                    drawCalls += 1; triangles += sphereIndexCount / 3
                case .box:
                    enc.setVertexBuffer(boxVerts, offset: 0, index: 0)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: boxVertCount)
                    drawCalls += 1; triangles += boxVertCount / 3
                }
            }
        }
        enc.endEncoding()
    }

    /// head-cam 渲染：256×256 rgba8，quackd render_headcam 语义——
    /// 针孔 fovy 90、黑底（无天空盒）、headlight gamma 朴素光照、无 AA/tonemap。
    /// viewProj/camPos/camFwd 由调用方从 head_camera 的 cam_xpos/cam_xmat 构建
    ///（MuJoCo 世界系，注意本路径不套 worldFromMujoco——item.model 直接用 MuJoCo 位姿）。
    func encodeHeadCamera(cmd: MTLCommandBuffer, items: [DrawItem],
                          viewProj: float4x4, camPos: SIMD3<Float>, camFwd: SIMD3<Float>) {
        if headDepthTexture == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: 256, height: 256, mipmapped: false)
            d.usage = [.renderTarget]
            d.storageMode = .private
            headDepthTexture = device.makeTexture(descriptor: d)
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = headTex
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.depthAttachment.texture = headDepthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setRenderPipelineState(psoRgba8)
        enc.setDepthStencilState(depthState)
        var vu = ViewUniforms(viewProj: viewProj, camPos: SIMD4(camPos, 1),
                              params: SIMD4(1, 0, 0, 0), camFwd: SIMD4(normalize(camFwd), 0))
        enc.setVertexBytes(&vu, length: MemoryLayout<ViewUniforms>.stride, index: 1)
        enc.setFragmentBytes(&vu, length: MemoryLayout<ViewUniforms>.stride, index: 1)
        for item in items {
            var iu = InstanceUniforms(model: item.model, color: item.rgba, misc: item.misc)
            enc.setVertexBytes(&iu, length: MemoryLayout<InstanceUniforms>.stride, index: 2)
            enc.setFragmentBytes(&iu, length: MemoryLayout<InstanceUniforms>.stride, index: 2)
            switch item.kind {
            case .meshSmooth(let g):
                enc.setVertexBuffer(g.smoothVertices, offset: 0, index: 0)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: g.smoothIndexCount,
                                          indexType: .uint32, indexBuffer: g.smoothIndices,
                                          indexBufferOffset: 0)
            case .meshFlat(let g):
                enc.setVertexBuffer(g.flatVertices, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: g.flatCount)
            case .plane:
                enc.setVertexBuffer(planeBuffer, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: planeVertCount)
            case .sphere:
                enc.setVertexBuffer(sphereVerts, offset: 0, index: 0)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: sphereIndexCount,
                                          indexType: .uint32, indexBuffer: sphereIndices,
                                          indexBufferOffset: 0)
            case .box:
                enc.setVertexBuffer(boxVerts, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: boxVertCount)
            }
        }
        enc.endEncoding()
    }

    /// 回读 head-cam 帧为 RGBA8 字节（行优先，原点在左上）。
    func readHeadPixels() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: 256 * 256 * 4)
        headTex.getBytes(&out, bytesPerRow: 256 * 4,
                         from: MTLRegionMake2D(0, 0, 256, 256), mipmapLevel: 0)
        return out
    }


    // 材质：albedo 当 sRGB 转线性，双方向光（主光+补光）+ 高光 + 菲涅尔，
    // 输出 HDR 线性，交给 QuiltRenderer 的 ACES tonemap（与参考 quilt 同管线）。
    // params.x=1（head-cam）：quackd render_headcam 语义——gamma 空间朴素光照
    //（MuJoCo OpenGL 渲染器不做线性化）：albedo·(headlight ambient .3 +
    // diffuse .6·max(dot(n,viewDir),0) + 顶光 .7·max(n.z,0))，直写 rgba8 无 tonemap。
    // 地板棋盘格：misc.w=1 蓝格 0.125m（quilt 默认），misc.w=2 灰格 0.1m
    //（quackd floor_cam，texrepeat 5/m → 0.1m 方格）。
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Vtx { packed_float3 pos; packed_float3 nrm; packed_float2 uv; };

    struct ViewUniforms {
        float4x4 viewProj;
        float4 camPos;
        float4 params;   // x: 0=quilt 线性管线, 1=quackd 头灯 gamma
        float4 camFwd;   // 相机前向（quackd headlight 方向光，恒定方向）
    };

    struct InstanceUniforms {
        float4x4 model;
        float4 color;
        float4 misc; // specular, shininess, emission, unused
    };

    struct VSOut {
        float4 pos [[position]];
        float3 nrm;
        float3 world;
        float2 uv;
        float2 local;   // geom 局部坐标（平面棋盘格用；单位四边形 [-1,1]）
    };

    vertex VSOut mjvVS(uint vid [[vertex_id]],
                       constant Vtx* verts [[buffer(0)]],
                       constant ViewUniforms& V [[buffer(1)]],
                       constant InstanceUniforms& I [[buffer(2)]]) {
        VSOut o;
        float4 wp = I.model * float4(verts[vid].pos, 1.0);
        o.pos = V.viewProj * wp;
        o.world = wp.xyz;
        o.nrm = (I.model * float4(verts[vid].nrm, 0.0)).xyz;
        o.uv = verts[vid].uv;
        o.local = verts[vid].pos.xy;
        return o;
    }

    fragment float4 mjvFS(VSOut in [[stage_in]],
                          constant ViewUniforms& V [[buffer(1)]],
                          constant InstanceUniforms& I [[buffer(2)]]) {
        float3 n = normalize(in.nrm);
        float3 albedoG = max(I.color.rgb, float3(0.0));   // gamma 空间（MuJoCo 朴素管线用）
        float3 albedo = pow(albedoG, float3(2.2));        // 线性（quilt HDR 管线用）
        // misc.w > 0.5 → 平面：可滤波棋盘格（iq 的 box-filtered checker）。
        // w=1 蓝格 0.125m；w=2 quackd 灰格 0.1m + mark 线；w=3 蓝格 + 竞技场边界描边
        if (I.misc.w > 0.5) {
            float sc = length(I.model[0].xyz);          // 单位四边形 → 米的缩放
            bool grayFloor = abs(I.misc.w - 2.0) < 0.5;
            float tile = grayFloor ? 0.1 : 0.125;
            float2 cm = in.local * sc / tile;
            float2 w = fwidth(cm) + 1e-5;
            float2 ii = 2.0 * (abs(fract(cm - 0.5 * w) - 0.5) - abs(fract(cm + 0.5 * w) - 0.5)) / w;
            float check = 0.5 - 0.5 * ii.x * ii.y;
            if (grayFloor) {
                // quackd floor_cam 灰格（gamma 值直接用）+ mark="edge" 0.85 网格线
                albedoG = mix(float3(0.50), float3(0.62), check);
                // 格边线：距方格边 <3% tile（带 fwidth 抗锯齿过渡）
                float2 dEdge = min(fract(cm), 1.0 - fract(cm));   // 单位：格
                float e = min(dEdge.x, dEdge.y);
                float ew = max(fwidth(cm.x), fwidth(cm.y)) + 1e-5;
                float mark = 1.0 - smoothstep(0.03 - ew, 0.03 + ew, e);
                albedoG = mix(albedoG, float3(0.85), mark);
            } else {
                // 固定深灰蓝双色调（参考 MuJoCo 默认地板），不吃 plane 自身的高亮 albedo
                albedoG = mix(float3(0.11, 0.13, 0.17), float3(0.20, 0.23, 0.28), check);
            }
            // 竞技场边界描边（w=3）：墙内侧 ±1.0m 处 1.2cm 灰线（墙色 0.55），
            // 平面化表示墙体边界——quilt 视角墙本体（group 4）不画，靠这条线示边界
            if (I.misc.w > 2.5) {
                float2 mp = in.local * sc;              // 米
                float db = min(abs(abs(mp.x) - 1.0), abs(abs(mp.y) - 1.0));
                float aa = max(fwidth(mp.x), fwidth(mp.y)) + 1e-5;
                float hl = max(0.012, aa);              // 半线宽（抗锯齿下限）
                float line = 1.0 - smoothstep(hl - aa, hl + aa, db);
                albedoG = mix(albedoG, float3(0.55, 0.55, 0.58), line);
            }
            albedo = pow(albedoG, float3(2.2));
        }
        if (V.params.x > 0.5) {
            // quackd 头灯：headlight 是沿相机前向的平行光（不是逐像素指向相机），
            // ambient .3 + diffuse .6·max(dot(n,camFwd),0) + 顶光 .7·max(n.z,0)
            float hl = max(dot(n, V.camFwd.xyz), 0.0);
            float sun = max(n.z, 0.0);
            float3 col = albedoG * (0.3 + 0.6 * hl + 0.7 * sun) + albedoG * I.misc.z;
            return float4(col, 1.0);
        }
        float3 L1 = normalize(float3(0.35, 0.75, 0.55));
        float3 L2 = normalize(float3(-0.6, 0.25, -0.45));
        float dif = max(dot(n, L1), 0.0);
        float dif2 = max(dot(n, L2), 0.0);
        float3 viewDir = normalize(V.camPos.xyz - in.world);
        float spec = pow(max(dot(reflect(-L1, n), viewDir), 0.0), I.misc.y * 60.0 + 8.0) * I.misc.x;
        float fre = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);
        float3 col = albedo * (0.18 + 0.80 * dif + 0.25 * dif2)
                   + spec * 0.4 + fre * albedo * 0.25 + albedo * I.misc.z;
        return float4(col, 1.0);
    }
    """
}
