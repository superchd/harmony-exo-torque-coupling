# File: harmony_space_recorder.py
import csv
import socket
import struct
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt

# PyQtGraph for fast real-time plotting
import pyqtgraph as pg

# =========================
# Save directory (fixed)
# =========================
SAVE_DIR = Path("/Users/hyundae/Desktop/harmonic-shr/Harmonic/records")

# =========================
# UDP Setup
# =========================
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # Non-blocking mode

# =========================
# Joint Info
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

NUM_VALUES = 28  # total doubles per packet

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Joint Angles and Torques")
        self.setFocusPolicy(Qt.StrongFocus)  # to receive SPACE key

        self.start_time = time.time()

        # ---- Rolling buffer for plotting ----
        self.time_buffer = deque(maxlen=2000)   # ~2000 samples
        self.value_buffer = deque(maxlen=2000)

        # ---- Recording state ----
        self.recording = False
        self.record_buffer = []  # (t, sh_abd_pos, sh_abd_trq, elb_pos, elb_trq)
        self.record_start_wallclock = None

        # ---- Main Layout ----
        outer_layout = QVBoxLayout()

        # Status bar
        status_layout = QHBoxLayout()
        self.rec_label = QLabel("○ IDLE  (Press SPACE to start)")
        self.rec_label.setStyleSheet("font-weight: bold; color: gray;")
        self.sample_count_label = QLabel("Samples: 0")
        status_layout.addWidget(self.rec_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.sample_count_label)
        outer_layout.addLayout(status_layout)

        # Scroll area for numeric labels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Joint", self))
        header.addWidget(QLabel("Value", self))
        scroll_layout.addLayout(header)

        # Build labels AND a mapping so each label gets the correct value index
        self.value_labels = []
        self.value_label_indices = []  # same length as value_labels

        # Helper to compute index in 'values' for a given label row
        # Order in 'values': Right Pos(0-6), Right Trq(7-13), Left Pos(14-20), Left Trq(21-27)
        def get_value_index(side: str, joint_idx: int, measure: str) -> int:
            if side == "Right":
                if measure.startswith("Position"):
                    return joint_idx  # 0..6
                else:  # Torque
                    return 7 + joint_idx  # 7..13
            else:  # Left
                if measure.startswith("Position"):
                    return 14 + joint_idx  # 14..20
                else:
                    return 21 + joint_idx  # 21..27

        # Add joint rows with correct mapping
        for side in ["Right", "Left"]:
            for j, joint in enumerate(joint_names):
                for measure in ["Position (rad)", "Torque (Nm)"]:
                    row = QHBoxLayout()
                    label_text = f"{side} {joint} - {measure}"
                    row.addWidget(QLabel(label_text, self))
                    value_label = QLabel("0.000", self)
                    row.addWidget(value_label)
                    scroll_layout.addLayout(row)

                    self.value_labels.append(value_label)
                    self.value_label_indices.append(get_value_index(side, j, measure))

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        # ---- Plot Widget ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Joint Value", units="")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        # default plot: right elbow flexion torque (index 12)
        self.joint_index_for_plot = IDX["R_ELB_FLEX_TRQ"]
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen('y', width=2))
        outer_layout.addWidget(self.plot_widget)

        self.setLayout(outer_layout)

        # ---- Timer for polling UDP data ----
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # ~200 Hz polling

    # -------------------------
    # Key press handler (SPACE toggles recording)
    # -------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if not self.recording:
                # Start recording
                self.recording = True
                self.record_buffer = []
                self.record_start_wallclock = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.rec_label.setText("● REC  (Press SPACE to stop)")
                self.rec_label.setStyleSheet("font-weight: bold; color: red;")
                self.sample_count_label.setText("Samples: 0")
                print("[INFO] Recording started.")
            else:
                # Stop recording and save
                self.recording = False
                self.rec_label.setText("○ IDLE  (Press SPACE to start)")
                self.rec_label.setStyleSheet("font-weight: bold; color: gray;")
                self.save_recording_to_csv()
            event.accept()
        else:
            super().keyPressEvent(event)

    # -------------------------
    # Save CSV
    # -------------------------
    def save_recording_to_csv(self):
        if not self.record_buffer:
            print("[WARN] No samples to save.")
            return

        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            fname = SAVE_DIR / f"harmony_record_{self.record_start_wallclock}.csv"
            header = [
                "t_sec",
                "r_shoulder_abduction_pos_rad",
                "r_shoulder_abduction_trq_Nm",
                "r_elbow_flexion_pos_rad",
                "r_elbow_flexion_trq_Nm",
            ]
            with open(str(fname), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.record_buffer)
            print(f"[INFO] Saved {len(self.record_buffer)} samples to {fname}")
            self.sample_count_label.setText(f"Samples: {len(self.record_buffer)} (saved)")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    # -------------------------
    # UDP polling
    # -------------------------
    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) == NUM_VALUES * 8:
                values = struct.unpack('28d', data)

                # Update numeric labels with correct mapping
                for label, idx in zip(self.value_labels, self.value_label_indices):
                    label.setText(f"{values[idx]:.3f}")

                # Plot selected channel
                selected_value = values[self.joint_index_for_plot]

                # Add time/value to buffers
                current_time = time.time() - self.start_time
                self.time_buffer.append(current_time)
                self.value_buffer.append(selected_value)

                # Update plot
                self.plot_curve.setData(list(self.time_buffer), list(self.value_buffer))

                # If recording, append selected joints to buffer
                if self.recording:
                    row = (
                        current_time,
                        values[IDX["R_SH_ABD_POS"]],
                        values[IDX["R_SH_ABD_TRQ"]],
                        values[IDX["R_ELB_FLEX_POS"]],
                        values[IDX["R_ELB_FLEX_TRQ"]],
                    )
                    self.record_buffer.append(row)
                    # light UI update
                    if len(self.record_buffer) % 20 == 0:
                        self.sample_count_label.setText(f"Samples: {len(self.record_buffer)}")

        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            print(f"Error receiving data: {e}")

# App entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.resize(900, 700)
    viewer.show()
    sys.exit(app.exec_())
