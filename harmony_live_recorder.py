#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmony SHR Live Recorder + (optional) Torque Individuation Index Analysis

Hotkeys:
  - SPACE : onset/offset toggle (mark movement window)
  - R     : start/stop recording (save CSV)
  - T     : set session tag to 'SABD' (file naming & analysis target)
  - M     : set session tag to 'EF_MVC' (file naming)
  - P     : cycle plotting channel (EF torque -> SABD torque -> SABD angle)

Behavior:
  - No dummy data generation. Uses only live UDP stream.
  - When you stop a recording:
      * CSV is saved under SAVE_DIR with columns:
        ['t_sec', 'r_shoulder_abduction_pos_rad', 'r_shoulder_abduction_trq_Nm',
         'r_elbow_flexion_pos_rad', 'r_elbow_flexion_trq_Nm', 'event_state', 'trigger']
      * IF session tag == 'SABD', it will try to find the latest EF_MVC file in SAVE_DIR.
        If found, runs T_SABD analysis and saves:
          <SABDbase>_individuation_summary.csv
          <SABDbase>_plots.png
        If not found, it prints a hint and skips analysis.

UDP packet:
  - 28 doubles (Right Pos 0-6, Right Trq 7-13, Left Pos 14-20, Left Trq 21-27)
"""

import sys, time, os, socket, struct
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QScrollArea)
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg

# =========================
# Paths & UDP setup
# =========================
SAVE_DIR = Path("/Users/hyundae/Desktop/harmonic-shr/Harmonic/records")

UDP_IP = "0.0.0.0"
UDP_PORT = 12345
NUM_VALUES = 28  # doubles
PACKET_BYTES = NUM_VALUES * 8

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# =========================
# Joint mapping
# =========================
joint_names = [
    "shoulder_elevation",
    "shoulder_protraction",
    "shoulder_abduction",
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "wrist_pronation"
]

# RIGHT angles: 0-6, RIGHT torques: 7-13, LEFT angles: 14-20, LEFT torques: 21-27
IDX = {
    "R_SH_ABD_POS": 2,            # right shoulder abduction position
    "R_SH_ABD_TRQ": 7 + 2,        # right shoulder abduction torque -> 9
    "R_ELB_FLEX_POS": 5,          # right elbow flexion position
    "R_ELB_FLEX_TRQ": 7 + 5,      # right elbow flexion torque -> 12
}

# =========================
# Analysis helpers
# =========================
def rad2deg(x):
    return np.rad2deg(x)

def moving_avg(x, win=11):
    if win < 2:
        return x
    return np.convolve(x, np.ones(win)/win, mode='same')

def robust_peak(x):
    """Robust magnitude: abs 95th percentile."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), 95))

def reps_from_triggers_df(df):
    """Make (start_idx, peak_idx_dummy, end_idx) from trigger/event_state."""
    starts, stops = [], []

    if "trigger" in df.columns:
        trig = df["trigger"].astype(str).str.lower()
        starts = df.index[trig == "onset"].tolist()
        stops  = df.index[trig == "offset"].tolist()

    if len(starts) == 0 and "event_state" in df.columns:
        st = df["event_state"].astype(str).str.lower()
        on = (st == "move").to_numpy()
        # rising edge as start (next index), falling as end (current)
        starts = (np.where((~on[:-1]) & (on[1:]))[0] + 1).tolist()
        stops  = (np.where((on[:-1]) & (~on[1:]))[0]).tolist()

    reps = []
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

def detect_reps_auto(t, shoulder_deg, min_peak_deg=70.0, start_thr_deg=10.0, min_gap_s=4.0):
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

def pick_latest_ef_mvc(path: Path) -> Path or None:
    cand = sorted(path.glob("harmony_record_*EF_MVC*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None

def analyze_T_index(sabd_csv_path: Path, ef_mvc_csv_path: Path, outdir: Path):
    """Compute T_SABD from SABD CSV + EF_MVC CSV. Save summary CSV and plot PNG."""
    sabd = pd.read_csv(sabd_csv_path)
    efmvc = pd.read_csv(ef_mvc_csv_path)

    # Required columns
    req_cols = [
        "t_sec",
        "r_shoulder_abduction_pos_rad",
        "r_shoulder_abduction_trq_Nm",
        "r_elbow_flexion_trq_Nm",
    ]
    missing = [c for c in req_cols if c not in sabd.columns]
    if missing:
        print(f"[WARN] SABD CSV missing columns: {missing}. Skip analysis.")
        return

    t = sabd["t_sec"].to_numpy()
    sh_pos_deg = rad2deg(sabd["r_shoulder_abduction_pos_rad"].to_numpy())
    sh_trq = sabd["r_shoulder_abduction_trq_Nm"].to_numpy()
    ef_trq = sabd["r_elbow_flexion_trq_Nm"].to_numpy()

    # rep windows by triggers first, else angle auto
    reps = reps_from_triggers_df(sabd)
    if len(reps) == 0:
        sh_pos_deg_s = moving_avg(sh_pos_deg, win=21)
        reps = detect_reps_auto(t, sh_pos_deg_s, min_peak_deg=70.0, start_thr_deg=10.0, min_gap_s=4.0)
    if len(reps) == 0:
        print("[WARN] No reps detected. Skip analysis.")
        return

    # refine peak idx (angle peak) within window
    reps_fixed = []
    for (s, _, e) in reps:
        if e <= s: 
            continue
        local_peak = int(np.argmax(sh_pos_deg[s:e+1])) + s
        reps_fixed.append((s, local_peak, e))
    reps = reps_fixed
    if len(reps) == 0:
        print("[WARN] Empty rep windows after refinement. Skip analysis.")
        return

    # EF MVC denom
    if "r_elbow_flexion_trq_Nm" not in efmvc.columns:
        print("[WARN] EF MVC CSV missing 'r_elbow_flexion_trq_Nm'. Skip analysis.")
        return
    ef_mvc_peak = robust_peak(efmvc["r_elbow_flexion_trq_Nm"].to_numpy())
    if ef_mvc_peak <= 1e-6:
        print("[WARN] EF MVC peak too small. Skip analysis.")
        return

    # per-rep T
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
            "dur_s": float(t[e]-t[s]),
            "SABD_peak_Nm": float(sabd_peak),
            "EF_peak_during_SABD_Nm": float(ef_peak_during_sabd),
            "EF_MVC_peak_Nm": float(ef_mvc_peak),
            "tau_bar_EF": float(tau_bar_EF),
            "T_SABD": float(T_sabd),
        })
    df_out = pd.DataFrame(rows)
    df_out["T_SABD_mean"] = df_out["T_SABD"].mean()
    df_out["T_SABD_std"]  = df_out["T_SABD"].std(ddof=1)

    outdir.mkdir(parents=True, exist_ok=True)
    base = sabd_csv_path.stem
    csv_out = outdir / f"{base}_individuation_summary.csv"
    png_out = outdir / f"{base}_plots.png"
    df_out.to_csv(csv_out, index=False)

    # Plot
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(t, sh_pos_deg, label="Shoulder ABD angle (deg)")
    ax1.plot(t, sh_trq, label="SABD torque (Nm)")
    ax1.plot(t, ef_trq, label="EF torque (Nm)")
    for (s, _, e) in reps:
        ax1.axvspan(t[s], t[e], alpha=0.1)
    ax1.set_xlabel("Time (s)"); ax1.set_title("SABD task traces"); ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(2,1,2)
    ax2.bar(df_out["rep"], df_out["T_SABD"])
    mean_T = df_out["T_SABD"].mean()
    ax2.axhline(mean_T, linestyle="--")
    ax2.set_ylim(-0.2, 1.05)
    ax2.set_xlabel("Rep"); ax2.set_ylabel("T_SABD")
    ax2.set_title(f"Torque Individuation Index (SABD)  |  mean={mean_T:.3f}")

    fig.tight_layout(); fig.savefig(png_out, dpi=150)
    print(f"[OK] Analysis saved CSV : {csv_out}")
    print(f"[OK] Analysis saved PNG : {png_out}")
    print(df_out[["rep","dur_s","T_SABD","tau_bar_EF","EF_peak_during_SABD_Nm"]])

# =========================
# Live viewer / recorder
# =========================
class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmony SHR — Live Joint Angles & Torques (REC + T-index)")
        self.setFocusPolicy(Qt.StrongFocus)  # to receive key events
        self.start_time = time.time()

        # state
        self.recording = False
        self.record_buffer = []  # rows
        self.record_start_stamp = None
        self.event_on = False
        self.pending_trigger = ""
        self.session_tag = "SABD"  # default tag in filename

        # plotting buffers
        self.time_buffer = deque(maxlen=2000)
        self.value_buffer = deque(maxlen=2000)
        self.plot_modes = [
            ("EF torque (Nm)", IDX["R_ELB_FLEX_TRQ"]),
            ("SABD torque (Nm)", IDX["R_SH_ABD_TRQ"]),
            ("SABD angle (rad)", IDX["R_SH_ABD_POS"]),
        ]
        self.plot_mode_idx = 0

        # ---- UI layout
        outer = QVBoxLayout()

        # status / controls
        top = QHBoxLayout()
        self.rec_label = QLabel("○ IDLE  (R=start, SPACE=on/off, T/M=tag, P=plot)")
        self.rec_label.setStyleSheet("font-weight: bold; color: gray;")
        self.sample_count_label = QLabel("Samples: 0")
        self.tag_label = QLabel(f"Tag: {self.session_tag}")
        self.plot_label = QLabel(f"Plot: {self.plot_modes[self.plot_mode_idx][0]}")
        top.addWidget(self.rec_label); top.addStretch(1)
        top.addWidget(self.tag_label); top.addSpacing(12)
        top.addWidget(self.plot_label); top.addSpacing(12)
        top.addWidget(self.sample_count_label)
        outer.addLayout(top)

        # scroll area for 28 values
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content)
        header = QHBoxLayout(); header.addWidget(QLabel("Joint")); header.addWidget(QLabel("Value"))
        scroll_layout.addLayout(header)
        self.value_labels = []
        for side in ["Right", "Left"]:
            for joint in joint_names:
                for measure in ["Position (rad)", "Torque (Nm)"]:
                    row = QHBoxLayout()
                    row.addWidget(QLabel(f"{side} {joint} - {measure}", self))
                    val = QLabel("0.000", self)
                    row.addWidget(val)
                    self.value_labels.append(val)
                    scroll_layout.addLayout(row)
        scroll_area.setWidget(scroll_content)
        outer.addWidget(scroll_area)

        # plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Joint Value", units="")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen('y', width=2))
        outer.addWidget(self.plot_widget)

        self.setLayout(outer)

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # ~200 Hz polling

        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- key events ----
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_R:
            if not self.recording:
                self._start_recording()
            else:
                self._stop_recording_and_save()
            event.accept(); return

        if key == Qt.Key_Space:
            if self.recording:
                self.event_on = not self.event_on
                self.pending_trigger = "onset" if self.event_on else "offset"
                state_str = "MOVE (on)" if self.event_on else "REST (off)"
                print(f"[MARK] {self.pending_trigger.upper()} @ {time.time()-self.start_time:.3f}s → {state_str}")
                self.sample_count_label.setText(f"Samples: {len(self.record_buffer)} | State: {state_str}")
            event.accept(); return

        if key == Qt.Key_T:
            self.session_tag = "SABD"
            self.tag_label.setText(f"Tag: {self.session_tag}")
            event.accept(); return

        if key == Qt.Key_M:
            self.session_tag = "EF_MVC"
            self.tag_label.setText(f"Tag: {self.session_tag}")
            event.accept(); return

        if key == Qt.Key_P:
            self.plot_mode_idx = (self.plot_mode_idx + 1) % len(self.plot_modes)
            self.plot_label.setText(f"Plot: {self.plot_modes[self.plot_mode_idx][0]}")
            event.accept(); return

        super().keyPressEvent(event)

    def _start_recording(self):
        self.recording = True
        self.record_buffer = []
        self.record_start_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_label.setText("● REC  (SPACE on/off, R stop)")
        self.rec_label.setStyleSheet("font-weight: bold; color: red;")
        self.sample_count_label.setText("Samples: 0")
        print("[INFO] Recording started. Tag:", self.session_tag)

    def _stop_recording_and_save(self):
        self.recording = False
        self.rec_label.setText("○ IDLE  (R=start, SPACE on/off, T/M tag, P plot)")
        self.rec_label.setStyleSheet("font-weight: bold; color: gray;")
        path = self._save_csv()
        print(f"[INFO] Saved CSV: {path}")
        # If this was SABD, try to analyze with latest EF_MVC
        if self.session_tag == "SABD":
            ef_path = pick_latest_ef_mvc(SAVE_DIR)
            if ef_path is None:
                print("[HINT] No EF_MVC CSV found in SAVE_DIR; analysis skipped.")
            else:
                try:
                    analyze_T_index(path, ef_path, SAVE_DIR)
                except Exception as e:
                    print(f"[WARN] Analysis failed: {e}")

    def _save_csv(self):
        if not self.record_buffer:
            print("[WARN] No samples to save.")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        tag = self.session_tag
        fname = SAVE_DIR / f"harmony_record_{tag}_{self.record_start_stamp}.csv"
        header = [
            "t_sec",
            "r_shoulder_abduction_pos_rad",
            "r_shoulder_abduction_trq_Nm",
            "r_elbow_flexion_pos_rad",
            "r_elbow_flexion_trq_Nm",
            "event_state",
            "trigger",
        ]
        import csv
        with open(str(fname), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.record_buffer)
        self.sample_count_label.setText(f"Samples: {len(self.record_buffer)} (saved)")
        return fname

    # ---- UDP polling ----
    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) == PACKET_BYTES:
                values = struct.unpack('28d', data)

                # update numeric labels (28 channels)
                for i in range(28):
                    self.value_labels[i].setText(f"{values[i]:.3f}")

                # pick channel to plot
                _, plot_idx = self.plot_modes[self.plot_mode_idx]
                selected_value = values[plot_idx]

                # add to buffers
                current_time = time.time() - self.start_time
                self.time_buffer.append(current_time)
                self.value_buffer.append(selected_value)

                # update plot
                self.plot_curve.setData(list(self.time_buffer), list(self.value_buffer))

                # if recording, append selected joints + state/trigger
                if self.recording:
                    state = "move" if self.event_on else "rest"
                    trig  = self.pending_trigger
                    row = (
                        current_time,
                        values[IDX["R_SH_ABD_POS"]],
                        values[IDX["R_SH_ABD_TRQ"]],
                        values[IDX["R_ELB_FLEX_POS"]],
                        values[IDX["R_ELB_FLEX_TRQ"]],
                        state,
                        trig,
                    )
                    self.record_buffer.append(row)
                    if len(self.record_buffer) % 20 == 0:
                        self.sample_count_label.setText(f"Samples: {len(self.record_buffer)}")
                    # trigger is single-shot
                    self.pending_trigger = ""

        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[ERR] receive_udp_data: {e}")

# ---- App entry point ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.resize(1000, 720)
    viewer.show()
    sys.exit(app.exec_())
