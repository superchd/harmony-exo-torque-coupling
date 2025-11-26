import socket
import struct
import sys
import os
import csv
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt

# ============================================================
# UDP Setup
# ============================================================
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # Non-blocking mode

# ============================================================
# Joint Info (for GUI 표시용)
# ============================================================
joint_names = [
    "shoulder_elevation",
    "shoulder_protraction",
    "shoulder_abduction",
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "wrist_pronation"
]

NUM_CHANNELS = 28  # 7 joints × 2 sides × (pos/torque)

# ============================================================
# 메인 위젯
# ============================================================
class JointDataViewer(QWidget):
    def __init__(self, subject_id: str, num_trials: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Joint Angles and Torques")
        self.setFocusPolicy(Qt.StrongFocus)  # 키보드 입력 받기 위해 필요

        self.subject_id = subject_id  # "H" 또는 "S"
        self.num_trials = num_trials  # 총 실험 횟수
        self.current_rep = 0          # 현재 rep index (1부터 시작)
        self.recording = False
        self.trial_start_time = None
        self.current_buffer = []      # 현재 trial의 데이터 (행 리스트)

        # ----- 세션 폴더 설정 -----
        base_dir = os.path.dirname(os.path.abspath(__file__))
        raw_root = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_root, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(raw_root, f"{self.subject_id}_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        # ----- GUI Layout -----
        self.value_labels = []
        outer_layout = QVBoxLayout()

        info_label = QLabel(
            f"Subject: {self.subject_id} | Trials: {self.num_trials} | "
            f"Session folder: {os.path.basename(self.session_dir)}"
        )
        outer_layout.addWidget(info_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Joint", self))
        header.addWidget(QLabel("Value", self))
        scroll_layout.addLayout(header)

        # Add joint rows
        for side in ["Right", "Left"]:
            for joint in joint_names:
                for measure in ["Position (degrees)", "Torque (Nm)"]:
                    row = QHBoxLayout()
                    label_text = f"{side} {joint} - {measure}"
                    row.addWidget(QLabel(label_text, self))
                    value_label = QLabel("0.000", self)
                    row.addWidget(value_label)
                    self.value_labels.append(value_label)
                    scroll_layout.addLayout(row)

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        # 상태 표시
        self.status_label = QLabel(
            "Press SPACE to start recording trial 1 / "
            f"{self.num_trials}. Press SPACE again to stop and save."
        )
        outer_layout.addWidget(self.status_label)

        self.setLayout(outer_layout)

        # Timer for polling UDP data
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # 5 ms for ~200 Hz polling

    # --------------------------------------------------------
    # 키보드 입력 (SPACE로 start/stop)
    # --------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_recording()
        else:
            super().keyPressEvent(event)

    def toggle_recording(self):
        # 이미 모든 trial 완료한 경우
        if not self.recording and self.current_rep >= self.num_trials:
            print("All trials completed. Exiting.")
            QApplication.quit()
            return

        # recording 시작
        if not self.recording:
            self.current_rep += 1
            self.current_buffer = []
            self.trial_start_time = time.perf_counter()
            self.recording = True
            msg = f"Started recording trial {self.current_rep}/{self.num_trials}"
            print(msg)
            self.status_label.setText(msg + " (Press SPACE to stop)")
        # recording 종료
        else:
            self.recording = False
            self.save_current_trial()
            msg = f"Stopped recording trial {self.current_rep}/{self.num_trials}"
            print(msg)
            if self.current_rep >= self.num_trials:
                self.status_label.setText(msg + " | All trials done. Exiting...")
                QApplication.quit()
            else:
                self.status_label.setText(
                    msg + f" | Press SPACE to start trial {self.current_rep+1}."
                )

    # --------------------------------------------------------
    # UDP 데이터 수신 + GUI 업데이트 + (옵션) 레코딩
    # --------------------------------------------------------
    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(2048)
        except BlockingIOError:
            return  # No data available
        except Exception as e:
            print(f"Error receiving data: {e}")
            return

        # 28 doubles (8 bytes each) = 224 bytes
        if len(data) != NUM_CHANNELS * 8:
            # 패킷 길이 안 맞으면 무시
            return

        try:
            values = struct.unpack('28d', data)
        except struct.error as e:
            print(f"Struct unpack error: {e}")
            return

        # GUI 업데이트 (지금처럼 rad -> deg 가정)
        for i in range(NUM_CHANNELS):
            # 여기서는 단순히 모두 rad->deg로 보여줌 (토크면 나중에 수정)
            self.value_labels[i].setText(f"{(values[i] * 57.2958):.3f}")

        # 레코딩 중이면 버퍼에 저장
        if self.recording:
            now = time.perf_counter()
            t_rel = now - self.trial_start_time if self.trial_start_time else 0.0

            # 행: [subject, rep_idx, t_rel, ch0..ch27]
            row = [self.subject_id, self.current_rep, t_rel] + list(values)
            self.current_buffer.append(row)

    # --------------------------------------------------------
    # 현재 trial 데이터를 CSV로 저장
    # --------------------------------------------------------
    def save_current_trial(self):
        if not self.current_buffer:
            print("No data to save for this trial.")
            return

        # 헤더 정의
        header = ["subject", "rep_idx", "t_rel_sec"] + [
            f"ch_{i}" for i in range(NUM_CHANNELS)
        ]

        filename = f"{self.subject_id}_rep{self.current_rep:02d}.csv"
        filepath = os.path.join(self.session_dir, filename)

        try:
            with open(filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.current_buffer)
            print(f"Saved trial {self.current_rep} data to: {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")


# ============================================================
# App entry point
# ============================================================
if __name__ == "__main__":
    # ---- 실험 메타데이터 입력 (콘솔) ----
    subject_id = input("Enter subject ID (H for healthy, S for stroke): ").strip().upper()
    while subject_id not in ("H", "S"):
        subject_id = input("Please enter 'H' or 'S': ").strip().upper()

    try:
        num_trials = int(input("Enter number of trials (e.g., 8): ").strip())
    except ValueError:
        print("Invalid number, defaulting to 8.")
        num_trials = 8

    app = QApplication(sys.argv)
    viewer = JointDataViewer(subject_id, num_trials)
    viewer.show()
    sys.exit(app.exec_())
