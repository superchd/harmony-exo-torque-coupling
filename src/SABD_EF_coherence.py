#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SABD_EF_coherence.py
- 폴더 안의 CSV(각 파일=1세트)를 일괄 처리하여
  저주파 코히어런스 대역평균(C_low)와 팔꿈치 표준편차를 요약/시각화.
- 요청사항 반영:
  1) CSV 패턴으로 파일을 읽도록 수정 (*.csv)
  2) 실패 행에도 동일 스키마 유지 (빈 값은 NaN + status에 이유 기록)
  3) 플롯 전에 방어 코드 (유효 값 없는 경우 KeyError 방지)

필수 CSV 컬럼(대소문자 무관 부분매칭 지원):
- t_rel
- Right_shoulder_abduction_degree
- Right_elbow_flexion_degree
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ==== 설정값 ====
DATA_DIR = Path("/Users/hyundae/Desktop/harmonic-shr/Harmonic/sets_csv")  # CSV가 들어있는 폴더
OUT_DIR  = DATA_DIR / "outputs"                                          # 결과 저장 폴더
OUT_DIR.mkdir(exist_ok=True)

THRESH  = 0.15         # 정상 상한선 τ
BAND    = (0.08, 0.6)  # 저주파 대역(Hz)
FS_TGT  = 20.0         # 다운샘플 타깃 Hz
TRIM_S  = 1.0          # 양끝 1초 제외(14초 녹화 → 가운데 12초 분석)
ELB_FLAT_STD = 0.5     # 팔꿈치 ~평평 기준 (deg)

# ==== SciPy 의존성 체크 ====
try:
    from scipy.signal import coherence, detrend, resample
except ImportError:
    sys.exit(
        "SciPy가 없습니다. 아래 명령으로 설치 후 다시 실행하세요.\n"
        "  python3 -m pip install --upgrade pip\n"
        "  python3 -m pip install scipy"
    )

# ==== 유틸 ====
def read_csv_auto(path: Path) -> pd.DataFrame:
    """첫 줄이 주석(#)이면 1줄 스킵, 아니면 그대로 CSV 로드."""
    with open(path, "r") as f:
        first = f.readline()
    return pd.read_csv(path, skiprows=1) if first.lstrip().startswith("#") else pd.read_csv(path)

def pick_column(df: pd.DataFrame, key: str):
    """
    대소문자 무관 부분매칭으로 열 이름 하나를 고른다.
    정확 일치 우선, 없으면 부분일치 중 첫 후보.
    """
    key_low = key.lower()
    exact = [c for c in df.columns if c.lower() == key_low]
    if exact:
        return exact[0]
    partial = [c for c in df.columns if key_low in c.lower()]
    return partial[0] if partial else None

def central_mask(t: np.ndarray, trim_s: float) -> np.ndarray:
    """시간축 t에서 양끝 trim_s를 버린 중앙 구간 마스크."""
    return (t >= t.min() + trim_s) & (t <= t.max() - trim_s)

def compute_clow_from_df(df: pd.DataFrame):
    """
    단일 세트 DataFrame에서 C_low와 팔꿈치 표준편차를 계산.
    실패 시 status 코드와 함께 NaN 반환(스키마 유지).
    """
    # 필수 컬럼 선택
    c_t   = pick_column(df, "t_rel")
    c_abd = pick_column(df, "Right_shoulder_abduction_degree")
    c_elb = pick_column(df, "Right_elbow_flexion_degree")
    if not (c_t and c_abd and c_elb):
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": "missing_required_columns"
        }

    try:
        D = df[[c_t, c_abd, c_elb]].dropna().astype(float).copy()
    except Exception as e:
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": f"astype_failed: {e}"
        }

    if D.empty or len(D) < 10:
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": "too_few_points"
        }

    t = D[c_t].to_numpy()
    x = D[c_abd].to_numpy()
    y = D[c_elb].to_numpy()

    # 중앙 구간(양끝 1초 제외)
    m = central_mask(t, TRIM_S)
    t, x, y = t[m], x[m], y[m]
    if len(t) < 40:  # 다운샘플 전 최소 길이 보장(대략)
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": "too_short_after_trim"
        }

    # 다운샘플(리샘플 → FS_TGT)
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.nanmedian(dt) <= 0:
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": "invalid_time_axis"
        }

    fs_est = 1.0 / float(np.nanmedian(dt))
    N = int(round(len(t) * FS_TGT / fs_est))
    if N < 32:  # 리샘플 결과가 너무 짧을 때
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": "too_short_after_resample"
        }

    try:
        x = resample(x, N)
        y = resample(y, N)
    except Exception as e:
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": np.nan,
            "flag_flat_elbow(<0.5deg)": np.nan,
            "below_thresh(τ=0.15)": np.nan,
            "status": f"resample_failed: {e}"
        }

    # detrend
    x = detrend(x, type="linear")
    y = detrend(y, type="linear")

    # 팔꿈치 거의 평평?
    elb_std = float(np.nanstd(y))
    if elb_std < ELB_FLAT_STD:
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": elb_std,
            "flag_flat_elbow(<0.5deg)": True,
            "below_thresh(τ=0.15)": True,  # flat이면 사실상 결합 없음으로 처리
            "status": "flat_elbow"
        }

    # 코히어런스
    try:
        f, Cxy = coherence(x, y, fs=FS_TGT, nperseg=64, noverlap=32)
    except Exception as e:
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": elb_std,
            "flag_flat_elbow(<0.5deg)": False,
            "below_thresh(τ=0.15)": np.nan,
            "status": f"coherence_failed: {e}"
        }

    band = (f >= BAND[0]) & (f <= BAND[1])
    if not np.any(band):
        return {
            "C_low_0.08_0.6Hz": np.nan,
            "elbow_std_deg": elb_std,
            "flag_flat_elbow(<0.5deg)": False,
            "below_thresh(τ=0.15)": np.nan,
            "status": "no_band_points"
        }

    C_low = float(np.nanmean(Cxy[band]))
    return {
        "C_low_0.08_0.6Hz": C_low,
        "elbow_std_deg": elb_std,
        "flag_flat_elbow(<0.5deg)": False,
        "below_thresh(τ=0.15)": (C_low < THRESH),
        "status": "ok"
    }

def empty_row(fname: str, reason: str):
    """실패해도 동일 스키마 유지."""
    return {
        "file": fname,
        "C_low_0.08_0.6Hz": np.nan,
        "elbow_std_deg": np.nan,
        "flag_flat_elbow(<0.5deg)": np.nan,
        "below_thresh(τ=0.15)": np.nan,
        "status": reason
    }

# ==== 메인 ====
def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print(f"[ERROR] CSV 파일이 없습니다: {DATA_DIR}")
        sys.exit(1)

    rows = []
    for f in files:
        try:
            df = read_csv_auto(f)
            res = compute_clow_from_df(df)
            res_row = {
                "file": f.name,
                **res
            }
            rows.append(res_row)
        except Exception as e:
            rows.append(empty_row(f.name, reason=f"load_failed: {e}"))

    summary = pd.DataFrame(rows)
    # 저장 (CSV는 항상 저장 / Excel은 엔진 있으면 저장)
    out_csv  = OUT_DIR / "coherence_summary.csv"
    out_xlsx = OUT_DIR / "coherence_summary.xlsx"
    summary.to_csv(out_csv, index=False)
    try:
        summary.to_excel(out_xlsx, index=False)  # openpyxl 미설치면 except로
        print("Saved:", out_xlsx)
    except Exception as e:
        print("[INFO] Excel 저장 실패 → CSV만 저장:", out_csv, "| 이유:", e)

    # 디버그용 간단 출력
    print("summary columns:", list(summary.columns))
    print(summary[["file", "status"]])

    # === 플롯 전 방어 코드 ===
    if "C_low_0.08_0.6Hz" not in summary.columns or summary["C_low_0.08_0.6Hz"].notna().sum() == 0:
        print("[WARN] 유효한 C_low 값이 없습니다. 입력 파일/컬럼명/길이/상태를 확인하세요.")
        return

    # 막대그래프 (flat 세트는 0으로 찍어두고, 텍스트 라벨 표시)
    vals = summary["C_low_0.08_0.6Hz"].to_numpy()
    labels = summary["file"].to_numpy()
    flats = summary["flag_flat_elbow(<0.5deg)"].fillna(False).to_numpy()

    plt.figure(figsize=(max(8, len(vals) * 0.9), 4.5))
    plt.bar(np.arange(len(vals)), np.nan_to_num(vals), tick_label=labels, width=0.6)
    plt.axhline(THRESH, linestyle="--", label=f"threshold τ={THRESH}")
    for i, is_flat in enumerate(flats):
        if is_flat:
            plt.text(i, 0.02, "flat elbow", ha="center", rotation=90)
    plt.ylim(0, 1.0)
    plt.ylabel("Coherence (mean in 0.08–0.6 Hz)")
    plt.title("Per-set low-frequency coherence (0.08–0.6 Hz)")
    plt.legend()
    plt.tight_layout()
    out_png = OUT_DIR / "coherence_bar.png"
    plt.savefig(out_png, dpi=150)
    print("Saved:", out_png)
    plt.close()

    # 간단 판정 요약
    valid = summary["below_thresh(τ=0.15)"]
    if valid.notna().any():
        rate_ok = float(np.nanmean(valid.astype(float))) * 100
        print(f"세트 중 τ 미만(또는 flat) 비율: {rate_ok:.1f}%")

if __name__ == "__main__":
    main()
