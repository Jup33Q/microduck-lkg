#!/usr/bin/env python3
"""BAM 对照 trace 的 Python 参考（infer_policy.py BAM 路径，一字不改地复用其
load_mujoco_with_bam / PolicyInference）。与 viewer 的 `--bam-trace out.csv`
跑同一场景（同初始姿态、同策略、同指令、同节拍），逐子步写同格式 CSV，
供 diff 做 bit 级移植验证。

用法（在 microduck_rl 仓库根，训练 venv）：
    UV_PROJECT_ENVIRONMENT=~/kimi-work/venvs/microduck_rl PYTHONPATH=$PWD/src \
    uv run python <本脚本> --policy <alpha_walking.onnx> --out /tmp/bam_ref.csv \
        [--ticks 200] [--cmd 0.3,0,0] [--no-bam]
"""

import argparse
import importlib.util
from pathlib import Path

import mujoco
import numpy as np

MICRODUCK_RL = Path(__file__).resolve().parents[2] / "microduck_rl"

spec = importlib.util.spec_from_file_location(
    "infer_policy", MICRODUCK_RL / "scripts" / "infer_policy.py"
)
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--cmd", default="0.3,0,0")
    ap.add_argument("--no-bam", action="store_true")
    args = ap.parse_args()

    xml = str(MICRODUCK_RL / ip.MICRODUCK_XML)
    if not args.no_bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
        model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
            xml, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    else:
        bam_ctrl = None
        model = mujoco.MjModel.from_xml_path(xml)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)

    pol = ip.PolicyInference(model, data, bam_ctrl=bam_ctrl,
                             walking_onnx_path=args.policy, new_cmd_obs=True,
                             use_projected_gravity=True)  # infer_policy 默认（--raw-accelerometer 的反面）= viewer 语义
    vx, vy, vyaw = (float(x) for x in args.cmd.split(","))
    pol.set_vel_cmd(vx, vy, vyaw)

    # 初始姿态 = viewer PolicyRunner.resetPose 的 keyframe 路径（STAND，缺省回退 key 0）
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, max(0, key))
    data.qvel[:] = 0.0
    if bam_ctrl is not None:
        data.ctrl[:] = 0.0          # viewer BAM resetPose：ctrl 由 update() 覆写
        bam_ctrl.reset(data.qpos)   # 清电压跌落状态，q_target = 当前关节位
    else:
        data.ctrl[:] = data.qpos[pol.joint_qpos_indices]
    mujoco.mj_forward(model, data)

    nq, nv, nu = model.nq, model.nv, model.nu
    # 受控 dof（与 viewer BamModelEdit.actuatedJoints 同一 passive_* 过滤）
    fric_dofs = [
        int(model.jnt_dofadr[model.actuator_trnid[i, 0]])
        for i in range(nu)
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  int(model.actuator_trnid[i, 0])) or "").startswith("passive_")
    ]
    header = (
        "step,time"
        + "".join(f",qpos{k}" for k in range(nq))
        + "".join(f",qvel{k}" for k in range(nv))
        + "".join(f",ctrl{k}" for k in range(nu))
        + "".join(f",fric{j}" for j in range(len(fric_dofs)))
    )
    lines = [header]
    step = 0
    for _ in range(args.ticks):
        action = pol.infer()
        pol.apply_action(action)
        for _ in range(4):  # decimation 4 @ 0.005s
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)
            row = f"{step},{data.time:.17g}"
            row += "".join(f",{v:.17g}" for v in data.qpos)
            row += "".join(f",{v:.17g}" for v in data.qvel)
            row += "".join(f",{v:.17g}" for v in data.ctrl)
            row += "".join(f",{model.dof_frictionloss[dof]:.17g}" for dof in fric_dofs)
            lines.append(row)
            step += 1

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"=== bam-trace ref OK: {args.out} ({step} substeps, "
          f"bam={not args.no_bam}, cmd={args.cmd}, policy={Path(args.policy).name})")


if __name__ == "__main__":
    main()
