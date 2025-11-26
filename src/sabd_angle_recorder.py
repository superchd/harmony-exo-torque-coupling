#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2-only Shoulder Abduction (Right) Angle — Simple Approx in Degrees
- Uses ONLY Right shoulder_abduction position (R2) for angle
- Z: zero, I: invert sign, 1/2/3: LPF 2/5/10 Hz, U: RAD<->DEG, SPACE: record CSV

Keys:
  SPACE : start/stop recording (CSV saved on stop)
  Z     : zero current filtered angle (make it 0°)
  I     : invert sign (make abduction positive)
  1/2/3 : LPF cutoff 2 / 5 / 10 Hz
  U     : input unit toggle (DEG <-> RAD)  [default DEG]
  P     : pause/resume plotting
"""

import sys, time, struct, csv
from math import pi
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtNetwork import QUdpSocket
import pyqtgraph as pg

# ============ Config ============
SAVE_DIR = Path("/Users/hyundae/Desktop/harmonic-shr/Harmonic/records")
UDP_PORT = 12345
PACKET_BYTES = 28 * 8
FMT = struct.Struct('28d')  # 28 doubles

IDX_R2_POS = 2   # Right shoulder_abduction position
IDX_R2_TRQ = 9   # Right shoulder_abduction torque

UNIT_DEFAULT = "DEG"   # your stream likely already in degrees; toggle with 'U'
FPS_PLOT   = 60        # plotting refresh
FPS_LABELS = 10        # label refresh
BUF_SEC    = 12        # seconds kept in ring
FS_TARGET  = 200       # for buffer sizing only

N = int(BUF_SEC * FS_TARGET)

# ========== LPF ==========
class OnePoleLPF:
    def __init__(self, fc_hz=5.0):
        self.fc = float(fc_hz); self.y = None
    def set_cutoff(self, fc_hz: float):
        self.fc = max(0.01, float(fc_hz))
    def reset(self, x0=None):
        self.y = x0
    def update(self, x, dt):
        if self.y is None or dt <= 0: self.y = x; return self.y
        tau = 1.0/(2.0*pi*self.fc); a = dt/(tau+dt)
        self.y = self.y + a*(x - self.y); return self.y

def rad2deg(x): return x * (180.0 / pi)

# ========== Main Widget ==========
class R2SABD(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("R2-only SABD Angle (deg)")
        self.setFocusPolicy(Qt.StrongFocus)

        # State
        self.unit = UNIT_DEFAULT  # "DEG" or "RAD"
        self.sign = 1.0           # invert with 'I'
        self.zero = 0.0           # zero with 'Z'
        self.lpf  = OnePoleLPF(5.0)
        self.prev_t = None
        self.prev_disp = None
        self.peak_abs = 0.0
        self.paused = False

        # Ring buffers
        self.t0 = time.perf_counter()
        self.head = 0
        self.tbuf   = np.zeros(N, dtype=np.float64)
        self.y_raw  = np.zeros(N, dtype=np.float32)  # deg (signed, zeroed)
        self.y_filt = np.zeros(N, dtype=np.float32)  # deg (zeroed, filtered)
        self.trqbuf = np.zeros(N, dtype=np.float32)

        # Recording
        self.recording = False
        self.rec_t0 = None
        self.rows = []

        # UI
        outer = QVBoxLayout()

        top = QHBoxLayout()
        self.lbl_status = QLabel("○ IDLE  (SPACE=rec, Z=zero, I=invert, 1/2/3 LPF, U=unit, P=pause)")
        self.lbl_status.setStyleSheet("font-weight:bold; color:gray;")
        self.lbl_unit = QLabel(f"Unit: {self.unit}")
        self.lbl_cut  = QLabel("LPF: 5 Hz")
        self.lbl_info = QLabel("")
        top.addWidget(self.lbl_status); top.addStretch(1)
        top.addWidget(self.lbl_unit); top.addSpacing(10)
        top.addWidget(self.lbl_cut);  top.addSpacing(10)
        top.addWidget(self.lbl_info)
        outer.addLayout(top)

        mid = QHBoxLayout()
        self.lbl_ang = QLabel("Angle (°): 0.00   [abs] 0.00")
        self.lbl_vel = QLabel("Vel (°/s): 0.00")
        self.lbl_peak= QLabel("Peak |°|: 0.0")
        self.lbl_trq = QLabel("Torque (Nm): 0.00")
        for w in (self.lbl_ang, self.lbl_vel, self.lbl_peak, self.lbl_trq):
            w.setStyleSheet("font-size:16px; font-weight:bold;")
        mid.addWidget(self.lbl_ang); mid.addSpacing(12)
        mid.addWidget(self.lbl_vel); mid.addSpacing(12)
        mid.addWidget(self.lbl_peak); mid.addSpacing(12)
        mid.addWidget(self.lbl_trq);  mid.addStretch(1)
        outer.addLayout(mid)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "SABD Angle (°)")
        self.plot.setLabel("bottom", "Time (s)")
        self.cur = self.plot.plot(np.array([]), np.array([]), pen=pg.mkPen('y', width=2))
        outer.addWidget(self.plot)

        self.setLayout(outer)

        # UDP: QUdpSocket (low-latency)
        self.sock = QUdpSocket(self)
        if not self.sock.bind(UDP_PORT, QUdpSocket.ShareAddress):
            print("[ERR] UDP bind failed. Port busy? Close other apps and retry.")
            sys.exit(1)
        self.sock.readyRead.connect(self.on_udp)

        # Timers
        self.timer_plot = QTimer(self); self.timer_plot.setInterval(int(1000/FPS_PLOT))
        self.timer_plot.timeout.connect(self.on_plot); self.timer_plot.start()
        self.timer_labels = QTimer(self); self.timer_labels.setInterval(int(1000/FPS_LABELS))
        self.timer_labels.timeout.connect(self.on_labels); self.timer_labels.start()

    # -------- UDP --------
    def on_udp(self):
        while self.sock.hasPendingDatagrams():
            sz = self.sock.pendingDatagramSize()
            if sz != PACKET_BYTES:
                self.sock.readDatagram(sz)
                continue
            ba = self.sock.receiveDatagram()
            vals = FMT.unpack(ba.data())

            raw_pos = float(vals[IDX_R2_POS])  # position (deg or rad)
            trq     = float(vals[IDX_R2_TRQ])  # Nm

            deg_in = (raw_pos if self.unit == "DEG" else rad2deg(raw_pos)) * self.sign

            now = time.perf_counter()
            dt  = 0.005 if self.prev_t is None else max(1e-4, now - self.prev_t)
            self.prev_t = now

            deg_f = self.lpf.update(deg_in, dt)
            disp  = deg_f - self.zero
            rawz  = deg_in - self.zero

            if self.prev_disp is None: vel = 0.0
            else: vel = (disp - self.prev_disp) / dt if dt > 0 else 0.0
            self.prev_disp = disp
            self.peak_abs = max(self.peak_abs, abs(disp))

            # ring write
            i = self.head
            self.tbuf[i]   = now - self.t0
            self.y_filt[i] = disp
            self.y_raw[i]  = rawz
            self.trqbuf[i] = trq
            self.head = (i + 1) % N

            # recording
            if self.recording and self.rec_t0 is not None:
                t_rec = now - self.rec_t0
                self.rows.append([
                    round(t_rec, 6),
                    float(deg_f), float(disp), float(vel),
                    float(trq),
                    self.unit, f"{self.lpf.fc:.2f}", "+" if self.sign>0 else "-"
                ])

    # -------- Plot / Labels --------
    def on_plot(self):
        if self.paused: return
        idx = self.head
        if idx == 0:
            t = self.tbuf; y = self.y_filt
        else:
            t = np.concatenate((self.tbuf[idx:], self.tbuf[:idx]))
            y = np.concatenate((self.y_filt[idx:], self.y_filt[:idx]))
        self.cur.setData(t, y)

    def on_labels(self):
        i = (self.head - 1) % N
        j = (i - 1) % N
        disp = float(self.y_filt[i])
        raw_abs = float(self.y_raw[i]) + self.zero
        trq  = float(self.trqbuf[i])
        dt = max(1e-4, self.tbuf[i] - self.tbuf[j])
        vel = (self.y_filt[i] - self.y_filt[j]) / dt
        self.lbl_ang.setText(f"Angle (°): {disp:7.2f}   [abs] {raw_abs:7.2f}")
        self.lbl_vel.setText(f"Vel (°/s): {vel:7.2f}")
        self.lbl_peak.setText(f"Peak |°|: {self.peak_abs:5.1f}")
        self.lbl_trq.setText(f"Torque (Nm): {trq:7.3f}")

    # -------- Keys --------
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space:
            if not self.recording:
                self.recording = True
                self.rec_t0 = time.perf_counter()
                self.rows = []
                self.lbl_status.setText("● REC  (SPACE=stop)")
                self.lbl_status.setStyleSheet("font-weight:bold; color:red;")
                self.lbl_info.setText("")
            else:
                self.recording = False
                self.lbl_status.setText("○ IDLE  (SPACE=rec, Z=zero, I=invert, 1/2/3 LPF, U=unit, P=pause)")
                self.lbl_status.setStyleSheet("font-weight:bold; color:gray;")
                self.save_csv()
            e.accept(); return

        if k == Qt.Key_Z:
            i = (self.head - 1) % N
            self.zero += float(self.y_filt[i])
            self.peak_abs = 0.0
            self.lbl_info.setText("Zero set.")
            e.accept(); return

        if k == Qt.Key_I:
            self.sign *= -1.0
            self.lbl_info.setText(f"Sign: {'+' if self.sign>0 else '-'}")
            e.accept(); return

        if k in (Qt.Key_1, Qt.Key_2, Qt.Key_3):
            fc = {Qt.Key_1:2.0, Qt.Key_2:5.0, Qt.Key_3:10.0}[k]
            self.lpf.set_cutoff(fc)
            self.lbl_cut.setText(f"LPF: {fc:.0f} Hz")
            e.accept(); return

        if k == Qt.Key_U:
            self.unit = "DEG" if self.unit == "RAD" else "RAD"
            self.lbl_unit.setText(f"Unit: {self.unit}")
            self.lbl_info.setText("Unit toggled.")
            e.accept(); return

        if k == Qt.Key_P:
            self.paused = not self.paused
            e.accept(); return

        super().keyPressEvent(e)

    # -------- Save --------
    def save_csv(self):
        if not self.rows:
            self.lbl_info.setText("No samples to save."); return
        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = SAVE_DIR / f"sabd_r2_only_{ts}.csv"
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_sec","angle_deg_abs","angle_deg_zeroed","vel_deg_s",
                            "sabd_trq_Nm","input_unit","lpf_hz","sign"])
                w.writerows(self.rows)
            self.lbl_info.setText(f"Saved: {fpath}")
        except Exception as ex:
            self.lbl_info.setText(f"Save failed: {ex}")

# ---------- main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = R2SABD()
    w.resize(960, 600)
    w.show()
    sys.exit(app.exec_())
