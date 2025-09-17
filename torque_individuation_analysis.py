#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torque Individuation Index (T_j) for SABD (n=2: SABD & EF)

- 파일 경로가 주어졌는데 없으면, 자동으로 더미 CSV(동적 SABD, EF MVC)를
  /Users/hyundae/Desktop/harmonic-shr/Harmonic/records 에 생성한 뒤 분석까지 수행합니다.
- 스페이스 마커(trigger=onset/offset, event_state=move/rest)가 있으면 우선 사용하고,
  없으면 어깨 각도로 자동 반복 구간을 검출합니다.

정의:
  T_SABD = 1 - (EF_peak_during_SABD / EF_peak_when_EF_instructed)

입력(선택):
  --sabd_csv   : 동적 SABD CSV (없거나 경로가 틀리면 자동 생성)
  --ef_mvc_csv : EF MVC CSV (없거나 경로가 틀리면 자동 생성)
  --outdir     : 결과 저장 폴더(미지정 시 기본 폴더 고정)

출력(항상 자동 저장):
  - <base>_individuation_summary.csv
  - <base>_plots.png
"""

import argparse
import math
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 고정 출력 경로 =========
OUTDIR_DEFAULT = "/Users/hyundae/Desktop/harmonic-shr/Harmonic/records"

# ========= 생성용 기본 설정 =========
FS       = 200.0   # Hz
DT       = 1.0/FS
NP_SEED  = 7

# SABD task (6 reps: ↑3s–hold1s–↓3s–rest6s, baseline 2s)
REPS             = 6
UP_S, HOLD_S, DOWN_S, REST_S = 3.0, 1.0, 3.0, 6.0
BASELINE_S       = 2.0
SABD_MAX_DEG     = 90.0
# torque model for SABD: T = K_angle*ang(rad) + K_vel*vel(rad/s) + noise
K_ANGLE, K_VEL   = 10.0, 0.5
# elbow coupling: T_ef = C * T_sabd + noise (rep마다 약간 변동)
COUPLING_MEAN, COUPLING_SD = 0.12, 0.03

# EF MVC (quiet1s + ramp0.5s + hold3s + relax0.5s + rest1s)
MVC_RAMP_S, MVC_HOLD_S, MVC_RELAX_S = 0.5, 3.0, 0.5
EF_MVC_PEAK_NM = 40.0

# ========= 유틸 =========
def rad2deg(x):
    return np.rad2deg(x)

def moving_avg(x, win=11):
    if win < 2:
        return x
    return np.convolve(x, np.ones(win)/win, mode='same')

def robust_peak(x):
    """절대값 95백분위수(노이즈에 강함)."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), 95))

def detect_reps_auto(t, shoulder_deg, min_peak_deg=70.0, start_thr_deg=10.0, min_gap_s=4.0):
    """어깨 각도로 0→90→0 반복 자동 검출."""
    reps, above, last_start_t = [], False, -1e9
    start_i = peak_i = None
    peak_val = -1e9
    for i, ang in enumerate(shoulder_deg):
        if not above and ang >= start_thr_deg:
            if t[i] - last_start_t >= min_gap_s:
                above, start_i, peak_i, peak_val = True, i, i, ang
                last_start_t = t[i]
        if above:
            if ang > peak_val:
                peak_val, peak_i = ang, i
            if ang < start_thr_deg and i > start_i:
                end_i = i
                if peak_val >= min_peak_deg:
                    reps.append((start_i, peak_i, end_i))
                above, start_i, peak_i, peak_val = False, None, None, -1e9
    return reps

def reps_from_triggers(df):
    """trigger('onset'/'offset') 또는 event_state(rest/move)로 반복 구간 구성."""
    reps = []

    starts, stops = [], []
    if "trigger" in df.columns:
        trig = df["trigger"].astype(str).str.lower()
        starts = df.index[trig == "onset"].tolist()
        stops  = df.index[trig == "offset"].tolist()

    if len(starts) == 0 and "event_state" in df.columns:
        st = df["event_state"].astype(str).str.lower()
        on = (st == "move").to_numpy()
        # rising edge는 다음 인덱스를 시작으로
        starts = (np.where((~on[:-1]) & (on[1:]))[0] + 1).tolist()
        # falling edge는 현재 인덱스를 종료로
        stops  = (np.where((on[:-1]) & (~on[1:]))[0]).tolist()

    si = 0
    for s in starts:
        while si < len(stops) and stops[si] <= s:
            si += 1
        if si < len(stops):
            reps.append((s, s, stops[si]))
            si += 1

    if len(starts) > len(stops) and len(df) > 0:
        s = starts[-1]
        reps.append((s, s, len(df)-1))

    return reps

# ========= 더미 데이터 생성기 =========
def _minjerk_profile(T, n):
    """0..1 min-jerk 프로파일."""
    if n <= 0: return np.array([])
    t = np.linspace(0.0, T, n, endpoint=False)
    tau = t / max(T, 1e-9)
    return 10*tau**3 - 15*tau**4 + 6*tau**5

def _add_noise(x, sd):
    return x + np.random.normal(0.0, sd, size=x.shape)

def _build_sabd_rep():
    """한 번의 0→90→0 동작(rad)."""
    n_up   = int(UP_S * FS)
    n_hold = int(HOLD_S * FS)
    n_down = int(DOWN_S * FS)
    up   = _minjerk_profile(UP_S, n_up) * math.radians(SABD_MAX_DEG)
    hold = np.ones(n_hold) * (up[-1] if up.size else 0.0)
    down = up[::-1][:n_down] if up.size else np.zeros(n_down)
    return np.concatenate([up, hold, down])

def generate_sabd_trial_df():
    """동적 SABD CSV 컬럼 그대로 생성."""
    np.random.seed(NP_SEED)
    # baseline
    n_base = int(BASELINE_S * FS)
    pos = [np.zeros(n_base)]
    event_state = ["rest"] * n_base
    trigger     = [""] * n_base
    onset_idx, offset_idx = [], []

    # reps
    for _ in range(REPS):
        rep_pos = _build_sabd_rep()
        n_rep   = rep_pos.size
        n_rest  = int(REST_S * FS)

        pos.append(rep_pos)
        event_state += ["move"] * n_rep
        trig = [""] * n_rep
        trig[0]  = "onset"
        trig[-1] = "offset"
        trigger += trig
        onset_idx.append(len(trigger)-n_rep)
        offset_idx.append(len(trigger)-1)

        pos.append(np.zeros(n_rest))
        event_state += ["rest"] * n_rest
        trigger     += [""] * n_rest

    pos = np.concatenate(pos)
    pos = _add_noise(pos, sd=math.radians(0.3))  # 각도 소량 노이즈

    # elbow angle: 거의 0, 동작 중 살짝 굴곡
    elb_pos = _add_noise(np.zeros_like(pos), sd=math.radians(0.5))
    for s, e in zip(onset_idx, offset_idx):
        L = max(e - s + 1, 1)
        elb_pos[s:e+1] += math.radians(5.0) * _minjerk_profile(L/FS, L)

    vel = np.gradient(pos, DT)
    sabd_trq = K_ANGLE * pos + K_VEL * vel
    sabd_trq = _add_noise(sabd_trq, sd=0.2)

    ef_trq = np.zeros_like(sabd_trq)
    for s, e in zip(onset_idx, offset_idx):
        c = float(np.clip(np.random.normal(COUPLING_MEAN, COUPLING_SD), 0.02, 0.25))
        ef_trq[s:e+1] = c * sabd_trq[s:e+1]
    ef_trq = _add_noise(ef_trq, sd=0.15)

    t = np.arange(pos.size) * DT

    return pd.DataFrame({
        "t_sec": t,
        "r_shoulder_abduction_pos_rad": pos,
        "r_shoulder_abduction_trq_Nm":  sabd_trq,
        "r_elbow_flexion_pos_rad":      elb_pos,
        "r_elbow_flexion_trq_Nm":       ef_trq,
        "event_state": event_state,
        "trigger": trigger,
    })

def generate_ef_mvc_df():
    """EF MVC CSV 생성."""
    np.random.seed(NP_SEED+1)

    def zeros_n(sec): return np.zeros(int(sec*FS))

    pieces, states, triggers = [], [], []

    # quiet 1s
    z = zeros_n(1.0); pieces.append(z); states += ["rest"] * z.size; triggers += [""] * z.size
    # ramp
    n_ramp = int(MVC_RAMP_S * FS)
    ramp = np.linspace(0, EF_MVC_PEAK_NM, n_ramp, endpoint=False)
    pieces.append(ramp); states += ["move"] * n_ramp; tri = [""] * n_ramp; tri[0]="onset"; triggers += tri
    # hold
    n_hold = int(MVC_HOLD_S * FS)
    hold = np.ones(n_hold) * EF_MVC_PEAK_NM
    pieces.append(hold); states += ["move"] * n_hold; triggers += [""] * n_hold
    # relax
    n_relax = int(MVC_RELAX_S * FS)
    relax = np.linspace(EF_MVC_PEAK_NM, 0, n_relax, endpoint=False)
    pieces.append(relax); states += ["move"] * n_relax; tri = [""] * n_relax; tri[-1]="offset"; triggers += tri
    # rest 1s
    z = zeros_n(1.0); pieces.append(z); states += ["rest"] * z.size; triggers += [""] * z.size

    ef_trq = np.concatenate(pieces)
    ef_trq = _add_noise(ef_trq, sd=0.3)

    n = ef_trq.size
    t = np.arange(n) * DT

    return pd.DataFrame({
        "t_sec": t,
        "r_shoulder_abduction_pos_rad": _add_noise(np.zeros(n), sd=math.radians(0.2)),
        "r_shoulder_abduction_trq_Nm":  _add_noise(np.zeros(n), sd=0.1),
        "r_elbow_flexion_pos_rad":      _add_noise(np.zeros(n), sd=math.radians(0.3)),
        "r_elbow_flexion_trq_Nm":       ef_trq,
        "event_state": states,
        "trigger": triggers,
    })

def ensure_csv(path_or_none, kind, outdir):
    """
    path_or_none: 주어진 경로(없을 수 있음)
    kind: 'sabd' 또는 'ef_mvc'
    outdir: 저장 폴더
    반환: 실제 존재하는 CSV 경로
    """
    os.makedirs(outdir, exist_ok=True)

    # 경로가 주어졌고 실제 존재하면 그대로 사용
    if path_or_none and os.path.exists(path_or_none):
        return path_or_none

    # 존재하지 않으면 자동 생성
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if kind == "sabd":
        df = generate_sabd_trial_df()
        # 주어진 파일명이 문자열로 들어왔지만 존재하지 않는 경우를 대비해,
        # 베이스 이름을 깔끔히 생성
        base = f"harmony_record_{now}.csv"
        save_path = os.path.join(outdir, base)
        df.to_csv(save_path, index=False)
        print(f"[AUTO] SABD CSV generated: {save_path}")
        return save_path
    elif kind == "ef_mvc":
        df = generate_ef_mvc_df()
        base = f"harmony_record_EF_MVC_{now}.csv"
        save_path = os.path.join(outdir, base)
        df.to_csv(save_path, index=False)
        print(f"[AUTO] EF MVC CSV generated: {save_path}")
        return save_path
    else:
        raise ValueError("kind must be 'sabd' or 'ef_mvc'")

# ========= 메인 =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sabd_csv", default=None, help="동적 SABD CSV 경로(없어도 OK, 자동 생성)")
    ap.add_argument("--ef_mvc_csv", default=None, help="EF MVC CSV 경로(없어도 OK, 자동 생성)")
    ap.add_argument("--outdir", default=None, help="결과 저장 폴더(미지정 시 기본 폴더)")
    args = ap.parse_args()

    outdir = args.outdir or OUTDIR_DEFAULT
    os.makedirs(outdir, exist_ok=True)

    # --- 입력 CSV 확보(없으면 자동 생성) ---
    sabd_csv = ensure_csv(args.sabd_csv, "sabd", outdir=OUTDIR_DEFAULT)
    ef_mvc_csv = ensure_csv(args.ef_mvc_csv, "ef_mvc", outdir=OUTDIR_DEFAULT)

    # --- 읽기 ---
    sabd = pd.read_csv(sabd_csv)
    efmvc = pd.read_csv(ef_mvc_csv)

    # --- 컬럼 검사 ---
    req_cols = [
        "t_sec",
        "r_shoulder_abduction_pos_rad",
        "r_shoulder_abduction_trq_Nm",
        "r_elbow_flexion_trq_Nm",
    ]
    missing = [c for c in req_cols if c not in sabd.columns]
    if missing:
        raise ValueError(f"SABD CSV missing columns: {missing}")

    t = sabd["t_sec"].to_numpy()
    sh_pos_deg = rad2deg(sabd["r_shoulder_abduction_pos_rad"].to_numpy())
    sh_trq = sabd["r_shoulder_abduction_trq_Nm"].to_numpy()
    ef_trq = sabd["r_elbow_flexion_trq_Nm"].to_numpy()

    # --- 반복 구간 결정: 트리거 우선, 없으면 자동 ---
    reps = reps_from_triggers(sabd)
    if len(reps) == 0:
        sh_pos_deg_s = moving_avg(sh_pos_deg, win=21)
        reps = detect_reps_auto(t, sh_pos_deg_s, min_peak_deg=70.0, start_thr_deg=10.0, min_gap_s=4.0)
    if len(reps) == 0:
        raise RuntimeError("Reps not found. Check triggers or angle thresholds.")

    # --- 구간 peak 보정(각도 최대 인덱스) ---
    reps_fixed = []
    for (s, _, e) in reps:
        if e <= s:
            continue
        local_peak = int(np.argmax(sh_pos_deg[s:e+1])) + s
        reps_fixed.append((s, local_peak, e))
    reps = reps_fixed
    if len(reps) == 0:
        raise RuntimeError("Rep windows were empty after refinement.")

    # --- EF MVC peak (정규화 분모) ---
    if "r_elbow_flexion_trq_Nm" not in efmvc.columns:
        raise ValueError("EF MVC CSV missing column: 'r_elbow_flexion_trq_Nm'")
    ef_mvc_peak = robust_peak(efmvc["r_elbow_flexion_trq_Nm"].to_numpy())
    if ef_mvc_peak <= 1e-6:
        raise RuntimeError("EF MVC peak too small or zero. Check EF MVC file/content.")

    # --- 반복별 계산 ---
    rows = []
    for k, (s, p, e) in enumerate(reps, 1):
        seg = slice(s, e+1)
        sabd_peak = robust_peak(sh_trq[seg])
        ef_peak_during_sabd = robust_peak(ef_trq[seg])

        tau_bar_EF = ef_peak_during_sabd / ef_mvc_peak
        T_sabd = 1.0 - tau_bar_EF

        rows.append({
            "rep": k,
            "t_start": float(t[s]),
            "t_end": float(t[e]),
            "dur_s": float(t[e] - t[s]),
            "SABD_peak_Nm": float(sabd_peak),
            "EF_peak_during_SABD_Nm": float(ef_peak_during_sabd),
            "EF_MVC_peak_Nm": float(ef_mvc_peak),
            "tau_bar_EF": float(tau_bar_EF),
            "T_SABD": float(T_sabd),
        })

    df_out = pd.DataFrame(rows)
    df_out["T_SABD_mean"] = df_out["T_SABD"].mean()
    df_out["T_SABD_std"]  = df_out["T_SABD"].std(ddof=1)

    # --- 출력 경로 ---
    base_in = os.path.splitext(os.path.basename(sabd_csv))[0]
    csv_out = os.path.join(outdir, f"{base_in}_individuation_summary.csv")
    png_out = os.path.join(outdir, f"{base_in}_plots.png")

    # --- 저장 ---
    df_out.to_csv(csv_out, index=False)

    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, sh_pos_deg, label="Shoulder ABD angle (deg)")
    ax1.plot(t, sh_trq, label="SABD torque (Nm)")
    ax1.plot(t, ef_trq, label="EF torque (Nm)")
    for (s, _, e) in reps:
        ax1.axvspan(t[s], t[e], alpha=0.1)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("SABD task traces")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.bar(df_out["rep"], df_out["T_SABD"])
    mean_T = df_out["T_SABD"].mean()
    ax2.axhline(mean_T, linestyle="--")
    ax2.set_ylim(-0.2, 1.05)
    ax2.set_xlabel("Rep")
    ax2.set_ylabel("T_SABD")
    ax2.set_title(f"Torque Individuation Index (SABD)  |  mean={mean_T:.3f}")

    fig.tight_layout()
    fig.savefig(png_out, dpi=150)

    print(f"[OK] Saved CSV : {csv_out}")
    print(f"[OK] Saved Plot: {png_out}")
    print(df_out[["rep", "dur_s", "T_SABD", "tau_bar_EF", "EF_peak_during_SABD_Nm"]])

if __name__ == "__main__":
    main()
