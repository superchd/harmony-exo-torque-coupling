#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SABD_EF_plot_3d_trajectory.py
- 각 CSV(1세트)에서 3D 궤적 플롯 생성
  x = Right_elbow_flexion_degree
  y = Right_shoulder_abduction_degree
  z = t_rel (시간)
- 전체 구간(FULL)과 중앙 12초(CENTER12s, 양끝 1초 제외) 모두 저장
- 시간에 따라 색상 그라디언트로 표현 (초단위)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys

# ===== 경로/옵션 =====
DATA_DIR = Path("/Users/hyundae/Desktop/harmonic-shr/Harmonic/sets_csv")
OUT_DIR  = DATA_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRIM_S = 1.0        # 중앙 12초: 양끝 1초 제외 (14초 녹화 가정)
DS_MAX_POINTS = 800 # 점이 너무 많으면 균등 다운샘플링(렌더링 안정)

# ===== CSV 유틸 =====
def read_csv_auto(path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        first = f.readline()
    return pd.read_csv(path, skiprows=1) if first.lstrip().startswith("#") else pd.read_csv(path)

def pick_column(df: pd.DataFrame, key: str):
    key_low = key.lower()
    exact = [c for c in df.columns if c.lower() == key_low]
    if exact:
        return exact[0]
    partial = [c for c in df.columns if key_low in c.lower()]
    return partial[0] if partial else None

def central_mask(t: np.ndarray, trim_s: float):
    return (t >= t.min() + trim_s) & (t <= t.max() - trim_s)

def maybe_downsample(t, x, y, max_points=DS_MAX_POINTS):
    n = len(t)
    if n <= max_points:
        return t, x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return t[idx], x[idx], y[idx]

# ===== 3D 플로터 =====
def plot_3d_trajectory(t, x_elb, y_sabd, title, out_path):
    """
    t: 시간 (z축)
    x_elb: Right_elbow_flexion_degree (x축)
    y_sabd: Right_shoulder_abduction_degree (y축)
    """
    # 다운샘플(렌더링 부담 완화)
    t, x_elb, y_sabd = maybe_downsample(t, x_elb, y_sabd, DS_MAX_POINTS)

    # 컬러를 시간에 따라 그라디언트로
    norm = Normalize(vmin=float(np.nanmin(t)), vmax=float(np.nanmax(t)))
    colors = plt.cm.viridis(norm(t))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    # 궤적: 점 + 라인. 점은 시간색, 라인은 옅은 회색
    ax.plot3D(x_elb, y_sabd, t, color='0.7', linewidth=1.0, alpha=0.8)
    ax.scatter3D(x_elb, y_sabd, t, c=t, cmap='viridis', s=6)

    ax.set_xlabel("Elbow flexion (deg)")
    ax.set_ylabel("Shoulder abduction (deg)")
    ax.set_zlabel("Time (s)")
    ax.set_title(title)

    # 컬러바(시간)
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Time (s)")

    # 보기 좋은 각도로 회전 (원하면 수정)
    ax.view_init(elev=22, azim=-60)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)

# ===== 메인 루프 =====
def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print(f"[ERROR] CSV 파일이 없습니다: {DATA_DIR}")
        sys.exit(1)

    for f in files:
        try:
            df = read_csv_auto(f)

            c_t    = pick_column(df, "t_rel")
            c_elb  = pick_column(df, "Right_elbow_flexion_degree")
            c_sabd = pick_column(df, "Right_shoulder_abduction_degree")
            if not (c_t and c_elb and c_sabd):
                print(f"[SKIP] 필수 컬럼 누락: {f.name} -> t_rel / Right_elbow_flexion_degree / Right_shoulder_abduction_degree 필요")
                continue

            D = df[[c_t, c_elb, c_sabd]].dropna().astype(float).copy()
            if D.empty:
                print(f"[SKIP] 빈 데이터: {f.name}")
                continue

            t_all   = D[c_t].to_numpy()
            x_elb   = D[c_elb].to_numpy()
            y_sabd  = D[c_sabd].to_numpy()

            # --- 1) 전체 구간 3D ---
            title_full = f"3D Trajectory (Elbow vs Shoulder vs Time) — {f.name} [FULL]"
            out_full   = OUT_DIR / f"traj3D_FULL_{f.stem}.png"
            plot_3d_trajectory(t_all, x_elb, y_sabd, title_full, out_full)

            # --- 2) 중앙 12초 3D ---
            m = central_mask(t_all, TRIM_S)
            if m.sum() > 5:
                t_c   = t_all[m]
                x_c   = x_elb[m]
                y_c   = y_sabd[m]
                title_c = f"3D Trajectory (Elbow vs Shoulder vs Time) — {f.name} [CENTER 12s]"
                out_c   = OUT_DIR / f"traj3D_CENTER12s_{f.stem}.png"
                plot_3d_trajectory(t_c, x_c, y_c, title_c, out_c)
            else:
                print(f"[NOTE] 중앙 12초가 짧아 생략: {f.name}")

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

if __name__ == "__main__":
    main()
