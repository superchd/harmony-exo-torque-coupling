#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("/Users/hyundae/Desktop/harmonic-shr/Harmonic/sets_csv")
OUT_DIR  = DATA_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRIM_S = 1.0                 # 양끝 1초 제외(14초→가운데 12초)
RATIO_THRESH = 0.10          # elbow/shoulder range ratio 기준
SLOPE_THRESH = 0.05          # |slope| < 0.05 deg/deg 기준
PTP_ELBOW_MAX = 2.0          # elbow 변동폭 절대 기준 (deg)
R_ABS_MAX = 0.2              # |r| < 0.2 기준

def read_csv_auto(p: Path) -> pd.DataFrame:
    with open(p, "r") as f:
        first = f.readline()
    return pd.read_csv(p, skiprows=1) if first.lstrip().startswith("#") else pd.read_csv(p)

def pick_column(df, key):
    k = key.lower()
    exact = [c for c in df.columns if c.lower()==k]
    if exact: return exact[0]
    part  = [c for c in df.columns if k in c.lower()]
    return part[0] if part else None

def central_mask(t, trim_s):
    return (t >= t.min()+trim_s) & (t <= t.max()-trim_s)

rows = []
files = sorted(DATA_DIR.glob("*.csv"))
for f in files:
    try:
        df = read_csv_auto(f)
        c_t   = pick_column(df, "t_rel")
        c_sbd = pick_column(df, "Right_shoulder_abduction_degree")
        c_elb = pick_column(df, "Right_elbow_flexion_degree")
        if not (c_t and c_sbd and c_elb):
            rows.append({"file": f.name, "status": "missing_columns"})
            continue

        D = df[[c_t, c_sbd, c_elb]].dropna().astype(float)
        if D.empty or len(D) < 10:
            rows.append({"file": f.name, "status": "too_few_points"})
            continue

        t = D[c_t].to_numpy()
        x = D[c_sbd].to_numpy()  # shoulder
        y = D[c_elb].to_numpy()  # elbow

        # 중앙 12초 사용
        m = central_mask(t, TRIM_S)
        if m.sum() >= 10:
            x = x[m]; y = y[m]; t = t[m]

        # 지표 계산
        ptp_sh = float(np.max(x) - np.min(x))
        ptp_el = float(np.max(y) - np.min(y))
        ratio  = float(ptp_el / ptp_sh) if ptp_sh > 0 else np.nan

        # 선형회귀: y = a + b x
        if len(x) >= 3 and np.std(x) > 1e-6:
            slope, intercept, r, p, stderr = stats.linregress(x, y)
        else:
            slope, r, p, stderr = np.nan, np.nan, np.nan, np.nan

        # 판정
        flags = []
        if not np.isnan(ptp_el) and ptp_el <= PTP_ELBOW_MAX:
            flags.append("elbow_ptp_small")
        if not np.isnan(ratio) and ratio <= RATIO_THRESH:
            flags.append("low_ratio")
        if not np.isnan(slope) and abs(slope) <= SLOPE_THRESH:
            flags.append("flat_slope")
        if not np.isnan(r) and abs(r) <= R_ABS_MAX:
            flags.append("low_corr")

        status = "ok" if flags else "review"

        rows.append({
            "file": f.name,
            "ptp_shoulder_deg": round(ptp_sh, 3),
            "ptp_elbow_deg": round(ptp_el, 3),
            "range_ratio_elbow_over_shoulder": round(ratio, 4) if np.isfinite(ratio) else np.nan,
            "slope_deg_per_deg": round(slope, 4) if np.isfinite(slope) else np.nan,
            "pearson_r": round(r, 4) if np.isfinite(r) else np.nan,
            "p_value_r": p if np.isfinite(p) else np.nan,
            "flags": ";".join(flags),
            "status": status
        })
    except Exception as e:
        rows.append({"file": f.name, "status": f"load_failed:{e}"})

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "elbow_stability_summary.csv", index=False)
print("Saved:", OUT_DIR / "elbow_stability_summary.csv")
print(summary[["file","ptp_shoulder_deg","ptp_elbow_deg","range_ratio_elbow_over_shoulder","slope_deg_per_deg","pearson_r","status"]])
