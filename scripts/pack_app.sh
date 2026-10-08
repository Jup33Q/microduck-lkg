#!/bin/bash
# P5: 打包 MicroDuck-LKG.app → ~/Desktop（双击免终端启动）。
# 内容：release 二进制 + libmujoco/libonnxruntime（@rpath → MacOS/ 同目录，
# 二进制自带 @executable_path rpath）+ Resources/{robot,scenes,policies}。
# 签名：ad-hoc（本机自用）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASKROOT="$(cd "$ROOT/.." && pwd)"
APP="MicroDuck-LKG"
OUT="$ROOT/build/$APP.app"

cd "$ROOT"
swift build -c release

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources/robot" \
         "$OUT/Contents/Resources/scenes" "$OUT/Contents/Resources/policies"

cp -X .build/release/mujoco-lkg-viewer "$OUT/Contents/MacOS/$APP"
cp -X .build/release/libmujoco.*.dylib Vendor/OnnxRuntime/libonnxruntime.*.dylib \
   "$OUT/Contents/MacOS/"

# 机器人模型（自包含 xml + mesh 资产）
cp -X "$TASKROOT/microduck_rl/src/mjlab_microduck/robot/microduck/robot_groundcontact.xml" \
   "$OUT/Contents/Resources/robot/"
cp -RX "$TASKROOT/microduck_rl/src/mjlab_microduck/robot/microduck/assets" \
   "$OUT/Contents/Resources/robot/"
# 场景（{{ROBOT_DIR}} 模板由运行时替换 → Resources/robot）
cp -X "$ROOT/scenes/arena_ball.xml" "$OUT/Contents/Resources/scenes/"
# 策略（pollen 官方 10 个 ONNX；manifest/README 不带）
cp -X "$ROOT"/models/pollen-policies/*.onnx "$OUT/Contents/Resources/policies/"

# 图标：assets/AppIcon-1024.png → iconset 全尺寸 → icns
ICONSET="$OUT/Contents/Resources/AppIcon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
    sips -z $s $s "$ROOT/assets/AppIcon-1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) "$ROOT/assets/AppIcon-1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$OUT/Contents/Resources/AppIcon.icns"
rm -rf "$ICONSET"

cat > "$OUT/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>MicroDuck-LKG</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleIdentifier</key><string>local.microduck-lkg</string>
    <key>CFBundleName</key><string>MicroDuck-LKG</string>
    <key>CFBundleDisplayName</key><string>MicroDuck LKG</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.5.0</string>
    <key>CFBundleVersion</key><string>5</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

codesign_ok() {
    codesign --force --deep --sign - "$1" >/dev/null 2>&1
}
# iCloud 同步域拷来的文件带 xattr（FileProvider 元数据，可能异步后补），codesign 会拒签
xattr -cr "$OUT" 2>/dev/null || true
if ! codesign_ok "$OUT"; then
    sleep 2
    xattr -cr "$OUT" 2>/dev/null || true
    codesign_ok "$OUT"
fi
# 全量替换（merge 会让 Finder 图标缓存滞留旧版本）
rm -rf ~/Desktop/"$APP.app"
cp -R "$OUT" ~/Desktop/
touch ~/Desktop/"$APP.app"
lsregister -f ~/Desktop/"$APP.app" 2>/dev/null \
    || /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f ~/Desktop/"$APP.app"
echo "=== packed: ~/Desktop/$APP.app ($(du -sh "$OUT" | cut -f1))"
