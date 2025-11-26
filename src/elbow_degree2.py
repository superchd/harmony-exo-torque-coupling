#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SABD_EF_plot_2d_xy.py
- 각 CSV(1세트)에 대해 2D 플롯 생성 (z축 없음)
  x = Right_shoulder_abduction_degree
  y = Right_elbow_flexion_degree
- FULL / CENTER12s(양끝 1초 제외) 두 버전 저장
- 시간은 색상으로만 표현(컬러바), 좌표축은 2D만 사용
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

TRIM_S = 1.0          # CENTER: 양끝 1초 제외
DS_MAX_POINTS = 2000  # 렌더링 안정용 다운샘플 상한

# ---- 축 스케일 옵션 ----
# y축(엘보) 범위를 x축(숄더)의 스팬에 맞추어 동일 스팬으로 설정
USE_MATCH_Y_TO_X_SPAN = True
PAD_FRAC = 0.05        # 축 패딩(%)
SPAN_FLOOR = 5.0       # 너무 작은 스팬 방지
SPAN_CEIL  = 180.0     # 과도한 스팬 제한

# (선택) y축(엘보) 고정 범위 강제 적용하고 싶으면 True 로
USE_FIXED_Y_ELBOW = False
Y_ELBOW_LIM = (-50.0, -20.0)

# (선택) x축(숄더) 고정 범위
USE_FIXED_X_SHOULDER = False
X_SHOULDER_LIM = (0.0, 30.0)

# ===== 유틸 =====
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

def clamp_span(span):
    return float(np.clip(span, SPAN_FLOOR, SPAN_CEIL))

def compute_xy_limits(x_sabd, y_elb):
    """옵션에 따라 x/y 축 한계를 계산."""
    # 기본: 데이터 min/max
    x_min, x_max = float(np.nanmin(x_sabd)), float(np.nanmax(x_sabd))
    y_min, y_max = float(np.nanmin(y_elb)),  float(np.nanmax(y_elb))

    # 고정범위 우선 적용(있을 때)
    if USE_FIXED_X_SHOULDER:
        x0, x1 = X_SHOULDER_LIM
        x_min, x_max = min(x0, x1), max(x0, x1)
    if USE_FIXED_Y_ELBOW:
        y0, y1 = Y_ELBOW_LIM
        y_min, y_max = min(y0, y1), max(y0, y1)

    # y축을 x 스팬과 동일하게 맞추는 옵션
    if USE_MATCH_Y_TO_X_SPAN and not USE_FIXED_Y_ELBOW:
        x_span = clamp_span(x_max - x_min)
        y_mean = float(np.nanmean(y_elb))
        pad_x = x_span * PAD_FRAC
        pad_y = x_span * PAD_FRAC
        x_lim = (x_min - pad_x, x_max + pad_x)
        y_lim = (y_mean - x_span/2.0 - pad_y, y_mean + x_span/2.0 + pad_y)
        return x_lim, y_lim

    # 아니면 기본 min/max + 패딩
    pad_x = max((x_max - x_min) * PAD_FRAC, 1e-6)
    pad_y = max((y_max - y_min) * PAD_FRAC, 1e-6)
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)

# ===== 2D 플로터 =====
def plot_2d_xy(t, x_sabd, y_elb, title, out_path):
    """2D: x=shoulder, y=elbow (시간은 색상으로만 표현)"""
    # 다운샘플
    t, x_sabd, y_elb = maybe_downsample(t, x_sabd, y_elb, DS_MAX_POINTS)

    # 유효성 체크
    if len(t) == 0 or len(x_sabd) == 0 or len(y_elb) == 0:
        print(f"[WARN] empty arrays → skip plot: {out_path.name}")
        return

    # 축 한계 계산
    xlim, ylim = compute_xy_limits(x_sabd, y_elb)

    # 시간 색상
    t_min, t_max = float(np.nanmin(t)), float(np.nanmax(t))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min == t_max:
        # 시간 축이 무의미하면 단색으로 플롯
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        ax.plot(x_sabd, y_elb, color='0.75', linewidth=1.0, alpha=0.9)
        ax.scatter(x_sabd, y_elb, s=8, alpha=0.9)
    else:
        norm = Normalize(vmin=t_min, vmax=t_max)
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        # 얇은 회색 선으로 궤적
        ax.plot(x_sabd, y_elb, color='0.75', linewidth=1.0, alpha=0.9)
        # 시간색 점 (mappable로 사용)
        sc = ax.scatter(x_sabd, y_elb, c=t, cmap='viridis', norm=norm, s=8, alpha=0.9)
        # 컬러바는 반드시 ax와 mappable을 명시
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label("Time (s)")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Shoulder abduction (deg)")
    ax.set_ylabel("Elbow flexion (deg)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    print("Saved:", out_path)


# ===== 메인 =====
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

            t_all  = D[c_t].to_numpy()
            x_sabd = D[c_sabd].to_numpy()   # x = shoulder
            y_elb  = D[c_elb].to_numpy()    # y = elbow

            # --- FULL ---
            title_full = f"2D Trajectory: Shoulder → Elbow — {f.name} [FULL]"
            out_full   = OUT_DIR / f"traj2D_full_{f.stem}.png"
            plot_2d_xy(t_all, x_sabd, y_elb, title_full, out_full)

            # --- CENTER 12s ---
            m = central_mask(t_all, TRIM_S)
            if m.sum() > 5:
                t_c = t_all[m]; x_c = x_sabd[m]; y_c = y_elb[m]
                title_c = f"2D Trajectory: Shoulder → Elbow — {f.name} [CENTER 12s]"
                out_c   = OUT_DIR / f"traj2D_center12s_{f.stem}.png"
                plot_2d_xy(t_c, x_c, y_c, title_c, out_c)
            else:
                print(f"[NOTE] 중앙 12초가 짧아 생략: {f.name}")

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

if __name__ == "__main__":
    main()
