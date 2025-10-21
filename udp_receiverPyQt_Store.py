import socket, struct, sys, os, csv, time
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
from PyQt5.QtCore import QTimer, Qt

# ====== UDP & Packet Format ======
UDP_IP, UDP_PORT = "0.0.0.0", 12345
PACKET_DOUBLES = 28
PACKET_BYTES = PACKET_DOUBLES * 8
UNPACK_FMT = "<28d"   # 송신이 리틀엔디안 double이라고 가정. 다르면 맞춰 변경!

# ====== Joint Names ======
joint_names = [
    "shoulder_elevation",
    "shoulder_protraction",
    "shoulder_abduction",
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "wrist_pronation"
]

RAD2DEG = 57.29577951308232

# ====== Recording Path ======
RECORD_DIR = "/Users/hyundae/Desktop/harmonic-shr/Harmonic/records"
os.makedirs(RECORD_DIR, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Joint Angles and Torques (Press SPACE to Start/Stop Recording)")
        self.value_labels = []
        self.rec_label = QLabel("REC: OFF")
        self.rec_label.setStyleSheet("font-weight: bold; color: #b00;")
        self.setFocusPolicy(Qt.StrongFocus)  # 키 입력을 받기 위해 필요

        # ====== UI Layout ======
        outer = QVBoxLayout(self)
        outer.addWidget(self.rec_label)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content)

        header = QHBoxLayout()
        header.addWidget(QLabel("Joint"))
        header.addWidget(QLabel("Value"))
        layout.addLayout(header)

        for side in ["Right","Left"]:
            for j in joint_names:
                # Position
                row_p = QHBoxLayout()
                row_p.addWidget(QLabel(f"{side} {j} - Position (deg)"))
                v_p = QLabel("0.000"); row_p.addWidget(v_p); layout.addLayout(row_p)
                self.value_labels.append(v_p)
                # Torque
                row_t = QHBoxLayout()
                row_t.addWidget(QLabel(f"{side} {j} - Torque (N·m)"))
                v_t = QLabel("0.000"); row_t.addWidget(v_t); layout.addLayout(row_t)
                self.value_labels.append(v_t)

        scroll.setWidget(content); outer.addWidget(scroll)

        # ====== UDP Poll Timer ======
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_udp)
        self.timer.start(10)  # 100 Hz

        # ====== Recording State ======
        self.is_recording = False
        self.rec_file = None
        self.rec_writer = None
        self.t0 = None  # recording start epoch

        # column names for CSV (angles only)
        self.angle_headers = self._build_angle_headers()

    def _build_angle_headers(self):
        headers = []
        for side in ["Right", "Left"]:
            for j in joint_names:
                headers.append(f"{side}_{j}_pos_deg")
        return headers

    def keyPressEvent(self, event):
        # SPACE: toggle recording
        if event.key() == Qt.Key_Space:
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()

    def _start_recording(self):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = os.path.join(RECORD_DIR, f"harmony_angles_{ts}.csv")
        try:
            self.rec_file = open(fname, "w", newline="")
            self.rec_writer = csv.writer(self.rec_file)
            # header
            self.rec_writer.writerow(["time_iso", "elapsed_s"] + self.angle_headers)
            self.is_recording = True
            self.t0 = time.time()
            self.rec_label.setText(f"REC: ON   → {fname}")
            self.rec_label.setStyleSheet("font-weight: bold; color: #0a0;")
            print(f"[REC] started: {fname}")
        except Exception as e:
            print(f"[REC] cannot start: {e}")

    def _stop_recording(self):
        try:
            if self.rec_file:
                self.rec_file.flush()
                self.rec_file.close()
        finally:
            self.rec_file = None
            self.rec_writer = None
            self.is_recording = False
            self.rec_label.setText("REC: OFF")
            self.rec_label.setStyleSheet("font-weight: bold; color: #b00;")
            print("[REC] stopped")

    def poll_udp(self):
        # 비블로킹: 도착한 패킷 모두 처리
        while True:
            try:
                data, _ = sock.recvfrom(2048)
            except BlockingIOError:
                break
            except Exception as e:
                print("recv error:", e); break

            if len(data) != PACKET_BYTES:
                continue

            try:
                vals = struct.unpack(UNPACK_FMT, data)
            except struct.error as e:
                print("unpack error:", e); continue

            # GUI 갱신
            for i, v in enumerate(vals):
                if (i % 2) == 0:   # position
                    self.value_labels[i].setText(f"{v * RAD2DEG:.3f}")
                else:              # torque
                    self.value_labels[i].setText(f"{v:.3f}")

            # ====== Recording (angles only, in degrees) ======
            if self.is_recording and self.rec_writer is not None:
                # 짝수 인덱스(0,2,4,...)가 포지션이라고 가정
                angle_deg = [vals[i] * RAD2DEG for i in range(0, PACKET_DOUBLES, 2)]
                now = time.time()
                row = [
                    datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                    f"{now - self.t0:.6f}",
                ] + [f"{a:.6f}" for a in angle_deg]
                try:
                    self.rec_writer.writerow(row)
                except Exception as e:
                    print(f"[REC] write error: {e}")
                    self._stop_recording()

    def closeEvent(self, event):
        # 창 닫힐 때 파일 안전 종료
        if self.is_recording:
            self._stop_recording()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = JointDataViewer(); w.show()
    sys.exit(app.exec_())
