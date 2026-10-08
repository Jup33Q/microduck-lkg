#!/usr/bin/env python3
"""对照 bam_trace 两份 CSV（Python 参考 vs Swift viewer），报告逐列误差。

用法：python3 bam_trace_diff.py ref.csv swift.csv [--tolerance 1e-9]
打印：总行数一致性、每列 max|Δ| 的全局最大值、最差 10 列、误差随时间的增长曲线
（每 50 子步的最大绝对误差），用于判断是 bit 级一致还是混沌放大。
"""

import argparse
import sys

import numpy as np


def load(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return header, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("swift")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args()

    h1, a = load(args.ref)
    h2, b = load(args.swift)
    if h1 != h2:
        print(f"HEADER MISMATCH:\n  ref   cols={len(h1)}\n  swift cols={len(h2)}")
        for i, (x, y) in enumerate(zip(h1, h2)):
            if x != y:
                print(f"  first diff at col {i}: {x!r} vs {y!r}")
                break
        sys.exit(1)
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH: ref {a.shape} vs swift {b.shape}")
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        print(f"  (truncated to {n} rows for the diff)")

    diff = np.abs(a - b)
    max_per_col = diff.max(axis=0)
    worst = np.argsort(max_per_col)[::-1][:10]
    gmax = max_per_col.max()
    print(f"rows={a.shape[0]} cols={a.shape[1]}  global max|Δ| = {gmax:.3e}")
    print("worst columns:")
    for c in worst:
        print(f"  {h1[c]:>10s}  max|Δ| = {max_per_col[c]:.3e}")
    # 误差增长曲线（qpos/qvel 列，每 50 子步窗口）
    state_cols = [i for i, h in enumerate(h1) if h.startswith(("qpos", "qvel"))]
    print("state max|Δ| over time (50-substep windows):")
    for lo in range(0, a.shape[0], 50):
        hi = min(lo + 50, a.shape[0])
        w = diff[lo:hi][:, state_cols].max()
        t = a[hi - 1, 1]
        print(f"  rows {lo:>4}-{hi - 1:<4} t={t:6.3f}s  max|Δ| = {w:.3e}")
    ok = gmax <= args.tolerance
    print(f"{'PASS' if ok else 'FAIL'} @ tolerance {args.tolerance:g}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
