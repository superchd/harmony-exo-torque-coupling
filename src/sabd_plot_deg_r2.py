#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Right Shoulder Abduction (R2) Angle in Degrees
- UDP 28 doubles 수신
- 플롯: R2 position -> deg 변환 후 플롯
- 입력 단위가 rad인지 deg인지에 따라 INPUT_UNIT만 바꿔주세요.
"""

import socket, struct, sys, time
from math import pi
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# ===== 설정 =====
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
PACKET_BYTES = 28 * 8
FMT = struct.Struct('28d')  # 28 doubles

INPUT_UNIT = "DEG"  # 스트림이 'rad'라면 "RAD"로 바꾸세요 ("DEG"면 변환 없음)

# 인덱스: 오른팔 각도 0-6, 토크 7-13
IDX_R2_POS = 2  # Right shoulder_abduction position (각도 채널)

def rad2deg(x): return x * (180.0 / pi)

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Right Shoulder Abduction Angle (deg) — R2")

        # UDP 소켓
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((UDP_IP, UDP_PORT))
        except OSError as e:
            print(f"[ERR] UDP bind 실패: {e}")
            sys.exit(1)
        self.sock.setblocking(False)

        self.start_time = time.time()

        # 상단 상태 라벨
        top = QHBoxLayout()
        self.unit_label = QLabel(f"Input Unit: {INPUT_UNIT}")
        self.val_label  = QLabel("Angle (°): 0.00")
        for w in (self.unit_label, self.val_label):
            w.setStyleSheet("font-weight: bold;")
        top.addWidget(self.unit_label); top.addStretch(1); top.addWidget(self.val_label)

        # 숫자 테이블(원본 28채널 그대로)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content)
        header = QHBoxLayout(); header.addWidget(QLabel("Channel")); header.addWidget(QLabel("Value (raw)"))
        scroll_layout.addLayout(header)
        self.value_labels = []
        for i in range(28):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"idx {i}", self))
            lab = QLabel("0.000"); row.addWidget(lab)
            self.value_labels.append(lab)
            scroll_layout.addLayout(row)
        scroll_area.setWidget(scroll_content)

        # 플롯
        self.time_buffer  = deque(maxlen=2000)
        self.value_buffer = deque(maxlen=2000)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Angle (°)")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen('y', width=2))

        # 메인 레이아웃
        outer = QVBoxLayout()
        outer.addLayout(top)
        outer.addWidget(scroll_area)
        outer.addWidget(self.plot_widget)
        self.setLayout(outer)

        # 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # ~200Hz

    def receive_udp_data(self):
        try:
            data, _ = self.sock.recvfrom(1024)
        except BlockingIOError:
            return
        except Exception as e:
            print(f"Error receiving data: {e}")
            return

        if len(data) != PACKET_BYTES:
            return

        vals = FMT.unpack(data)

        # 28채널 숫자 그대로 표시(원시값 단위)
        for i in range(28):
            self.value_labels[i].setText(f"{vals[i]:.3f}")

        # R2 각도 -> deg
        r2_pos = float(vals[IDX_R2_POS])
        angle_deg = rad2deg(r2_pos) if INPUT_UNIT.upper() == "RAD" else r2_pos

        # 상단 라벨
        self.val_label.setText(f"Angle (°): {angle_deg:7.2f}")

        # 플롯 업데이트
        t = time.time() - self.start_time
        self.time_buffer.append(t)
        self.value_buffer.append(angle_deg)
        self.plot_curve.setData(list(self.time_buffer), list(self.value_buffer))

# ---- 실행 ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.resize(900, 700)
    viewer.show()
    sys.exit(app.exec_())
