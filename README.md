# MicroDuck-LKG

MicroDuck 双足机器人的 MuJoCo 仿真 viewer：Swift + Metal 原生 macOS app，支持 Looking Glass 光场显示器（quilt 渲染），内置 laya VLM 视觉决策（找球-走近-踢球）+ rsl_rl PPO 行走策略的两级控制，以及从 SwiftUI 面板一键发起本机 RL 训练。

架构与训练细节见 [roadmap.md](roadmap.md) 与 [P3-VLM-fusion-activation.md](P3-VLM-fusion-activation.md)。

## 模型权重（不入库，从 Hugging Face 下载）

| 用途 | HF 仓库 | 放置路径 |
|---|---|---|
| laya VLM 决策模型 | [`thaitea/laya-vision-microduck-kick`](https://huggingface.co/thaitea/laya-vision-microduck-kick) | `models/laya-vision-microduck-kick/` |
| 行走策略 ×10（含默认 `alpha_walking.onnx`） | [`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies) | `models/pollen-policies/` |
| 可续训的 base walk checkpoint | [`tfrere/microduck-move-base-walk`](https://huggingface.co/tfrere/microduck-move-base-walk) | `models/tfrere-base-walk/` |
| 本机自训 m1 系列行走策略 | [`Jup33QE/microduck-m1-policies`](https://huggingface.co/Jup33QE/microduck-m1-policies) | `../microduck_rl/`（仓库根） |

下载命令示例：

```bash
hf download thaitea/laya-vision-microduck-kick --local-dir models/laya-vision-microduck-kick
hf download pollen-robotics/microduck-policies --local-dir models/pollen-policies
hf download tfrere/microduck-move-base-walk --local-dir models/tfrere-base-walk
```

## 构建

需要 macOS + Xcode / Swift toolchain。第三方依赖（MuJoCo.xcframework、ONNX Runtime）已 vendor 在 `Vendor/` 下。

```bash
swift build            # 或参考 scripts/pack_app.sh 打包 .app
```

## 相关仓库

- [Jup33Q/microduck_rl](https://github.com/Jup33Q/microduck_rl) — RL 行走策略训练（fork 自 pollen-robotics，含 macOS CPU 训练支持）
- 上游：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
