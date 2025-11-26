import socket
import struct
import sys
import time
import math
import csv
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt

# New: PyQtGraph for fast real-time plotting
import pyqtgraph as pg

# UDP Setup
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # Non-blocking mode

# Joint Info
joint_names = [
    "shoulder_elevation",
    "shoulder_protraction",
    "shoulder_abduction",
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "wrist_pronation"
]

# Mapping (참고):
# RIGHT angles 0-6  -> [SE, SP, SABD, SR, SF, EF, WP]
# RIGHT torques 7-13
# LEFT  angles 14-20
# LEFT  torques 21-27
SABD_RIGHT_ANGLE_IDX = 2  # 오른팔 shoulder abduction 각도

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Joint Angles and Torques")
        self.setFocusPolicy(Qt.StrongFocus)  # 키 입력 받기 위해 포커스 정책 설정

        self.value_labels = []
        self.start_time = time.time()

        # ---- Recording state ----
        self.is_recording = False
        self.record_buffer = []  # (t_rel_s, sabd_deg)
        self.record_start_wall = None

        # ---- Rolling buffer for plotting ----
        self.max_history_sec = 10.0
        self.time_buffer = deque(maxlen=2000)  # Store ~2000 samples
        self.value_buffer = deque(maxlen=2000)

        # ---- Main Layout ----
        outer_layout = QVBoxLayout()

        # Status/Help row
        status_row = QHBoxLayout()
        self.rec_label = QLabel(" ")  # 녹화 상태 표기
        self.help_label = QLabel("Press 'R' to Start/Stop recording SABD (deg)")
        status_row.addWidget(self.rec_label)
        status_row.addStretch(1)
        status_row.addWidget(self.help_label)
        outer_layout.addLayout(status_row)

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

        # Add joint rows (값 라벨은 rad/Nm 그대로 표시)
        for side in ["Right", "Left"]:
            for joint in joint_names:
                for measure in ["Position (rad)", "Torque (Nm)"]:
                    row = QHBoxLayout()
                    label_text = f"{side} {joint} - {measure}"
                    row.addWidget(QLabel(label_text, self))
                    value_label = QLabel("0.000", self)
                    row.addWidget(value_label)
                    self.value_labels.append(value_label)
                    scroll_layout.addLayout(row)

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        # ---- Plot Widget ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Shoulder Abduction (deg)", units="")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen('y', width=2))
        outer_layout.addWidget(self.plot_widget)

        self.setLayout(outer_layout)

        # ---- Timer for polling UDP data ----
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # ~200 Hz polling

    def keyPressEvent(self, event):
        # R 키로 녹화 토글
        if event.key() in (Qt.Key_R,):
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_and_save_recording()
        super().keyPressEvent(event)

    def start_recording(self):
        self.is_recording = True
        self.record_buffer = []
        self.record_start_wall = time.time()
        self.rec_label.setText("● REC")
        self.rec_label.setStyleSheet("color: red; font-weight: bold;")
        print("[REC] Recording started")

    def stop_and_save_recording(self):
        self.is_recording = False
        self.rec_label.setText(" ")
        self.rec_label.setStyleSheet("")
        print("[REC] Recording stopped")

        if not self.record_buffer:
            print("[REC] No samples captured; nothing to save")
            return

        # 파일 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SABD_right_deg_{ts}.csv"
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "sabd_deg"])  # 헤더
                writer.writerows(self.record_buffer)
            print(f"[REC] Saved: {filename} ({len(self.record_buffer)} samples)")
        except Exception as e:
            print(f"[REC] Error saving CSV: {e}")

    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) == 28 * 8:
                values = struct.unpack('28d', data)

                # Update numeric labels (rad/Nm 그대로)
                for i in range(28):
                    self.value_labels[i].setText(f"{values[i]:.3f}")

                # 오른팔 SABD 각도 -> degree 변환
                sabd_deg = math.degrees(values[SABD_RIGHT_ANGLE_IDX])

                # 시간/플롯 업데이트
                current_time = time.time() - self.start_time
                self.time_buffer.append(current_time)
                self.value_buffer.append(sabd_deg)
                self.plot_curve.setData(list(self.time_buffer), list(self.value_buffer))

                # 녹화 중이면 버퍼에 추가 (녹화 시작 기준 상대시간)
                if self.is_recording and self.record_start_wall is not None:
                    t_rel = time.time() - self.record_start_wall
                    self.record_buffer.append((t_rel, sabd_deg))

        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            print(f"Error receiving data: {e}")

# App entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.resize(900, 650)
    viewer.show()
    sys.exit(app.exec_())
