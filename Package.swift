// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "mujoco-lkg",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "mujoco-lkg-probe", targets: ["mujoco-lkg-probe"]),
        .executable(name: "mujoco-lkg-viewer", targets: ["mujoco-lkg-viewer"])
    ],
    dependencies: [
        // StreamQuilt 本地源码依赖：QuiltMath/QuiltSpec/QuiltRenderer/Calibration
        // 全部由此复用，校准与 quilt 常量禁止在本 package 另抄一份。
        .package(path: "/Users/jup33q/Desktop/StreamQuilt")
    ],
    targets: [
        // libmujoco 3.10.0（microduck_rl venv wheel 自带，universal arm64+x86_64）。
        // install name 已修正为 @rpath/libmujoco.3.10.0.dylib。
        // 升级到 brew 3.15.0 时重新生成此 xcframework。
        .binaryTarget(name: "MuJoCo", path: "Vendor/MuJoCo.xcframework"),
        // onnxruntime 1.24.4（microduck_rl venv wheel 自带）→ 策略 ONNX 同运行时推理。
        // 不走 xcframework（两个 -library xcframework 的 module.modulemap 会在
        // include/ 撞名）；C shim target + -L/rpath 指到 Vendor 里的 dylib 副本。
        .target(
            name: "COnnxRuntime",
            path: "Sources/COnnxRuntime",
            publicHeadersPath: "include",
            linkerSettings: [
                .unsafeFlags([
                    "-L", "Vendor/OnnxRuntime",
                    "-Xlinker", "-rpath", "-Xlinker",
                    "/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360/mujoco-lkg/Vendor/OnnxRuntime",
                ])
            ]
        ),
        .executableTarget(
            name: "mujoco-lkg-probe",
            dependencies: ["MuJoCo"],
            // SPM 会把 xcframework 里的 dylib 拷到可执行文件旁，但 dev 构建不自动
            // 加 @executable_path rpath，这里补上以匹配 @rpath/libmujoco.3.10.0.dylib。
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"])]
        ),
        .executableTarget(
            name: "mujoco-lkg-viewer",
            dependencies: [
                "MuJoCo",
                "COnnxRuntime",
                .product(name: "StreamQuilt", package: "StreamQuilt")
            ],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"])]
        )
    ]
)
