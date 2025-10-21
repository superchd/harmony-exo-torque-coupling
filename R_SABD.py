import socket
import struct
import sys
import time
import math
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# ===== Settings =====
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
ENDIAN = "<"          # "<"=little-endian, "!"=network big-endian
NUM_DOUBLES = 28
PACKET_BYTES = NUM_DOUBLES * 8
PLOT_HISTORY_SEC = 10.0
# Right shoulder_abduction Position index (grouped 가정: 2)  interleaved면 4로 변경
IDX_POS = 2
# ====================

class RightSABDPosDegViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Right Shoulder Abduction - Position (deg) Live")
        self.start_time = time.time()

        self.t_buf  = deque(maxlen=4000)
        self.deg_buf = deque(maxlen=4000)

        # ---- UI ----
        layout = QVBoxLayout()

        status = QHBoxLayout()
        status.addWidget(QLabel(f"UDP {UDP_IP}:{UDP_PORT}"))
        status.addWidget(QLabel(f"Index (pos) = {IDX_POS}"))
        status.addStretch(1)
        layout.addLayout(status)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Position", units="deg")
        self.plot.setLabel("bottom", "Time", units="s")
        self.curve_deg = self.plot.plot([], [], pen=pg.mkPen('y', width=2), name="deg")
        layout.addWidget(self.plot)

        row2 = QHBoxLayout()
        self.lbl_deg = QLabel("Pos: 0.00 °")
        row2.addWidget(self.lbl_deg)
        row2.addStretch(1)
        layout.addLayout(row2)

        self.setLayout(layout)

        # UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(5)  # ~200 Hz

    def tick(self):
        try:
            data, _ = self.sock.recvfrom(1024)
        except BlockingIOError:
            return
        except Exception as e:
            print("recv error:", e)
            return

        if len(data) != PACKET_BYTES:
            return

        try:
            vals = struct.unpack(f"{ENDIAN}{NUM_DOUBLES}d", data)
        except Exception as e:
            print("unpack error:", e)
            return

        rad = vals[IDX_POS]
        deg = rad * (180.0 / math.pi)
        t = time.time() - self.start_time

        self.t_buf.append(t)
        self.deg_buf.append(deg)

        # keep last N seconds
        while self.t_buf and (t - self.t_buf[0]) > PLOT_HISTORY_SEC:
            self.t_buf.popleft()
            self.deg_buf.popleft()

        t_list = list(self.t_buf)
        self.curve_deg.setData(t_list, list(self.deg_buf))
        self.lbl_deg.setText(f"Pos: {deg:.2f} °")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RightSABDPosDegViewer()
    w.resize(900, 500)
    w.show()
    sys.exit(app.exec_())
