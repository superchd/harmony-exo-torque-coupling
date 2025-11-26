import socket
import struct
import sys
import time
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import QTimer

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

NUM_JOINTS = len(joint_names) * 2 * 2  # Right/Left * pos/torque

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Joint Angles and Torques")

        self.value_labels = []
        self.start_time = time.time()

        # ---- Rolling buffer for plotting ----
        self.max_history_sec = 10.0
        self.time_buffer = deque(maxlen=2000)  # Store ~2000 samples
        self.value_buffer = deque(maxlen=2000)

        # ---- Main Layout ----
        outer_layout = QVBoxLayout()

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

        # Add joint rows
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
        self.plot_widget.setLabel("left", "Joint Value", units="")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen('y', width=2))
        outer_layout.addWidget(self.plot_widget)

        self.setLayout(outer_layout)

        # ---- Timer for polling UDP data ----
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # ~200 Hz polling

    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) == 28 * 8:
                values = struct.unpack('28d', data)

                # Update numeric labels
                for i in range(28):
                    self.value_labels[i].setText(f"{values[i]:.3f}")

                # 0SE, 1SP, 2SABD, 3SR, 4SF, 5EF, 6WP
                # RIGHT angles 0-6 RIGHT trq 7-13 LEFT angles 14-20 LEFT trq 21-27
                joint_index_for_plot = 13
                selected_value = values[joint_index_for_plot]

                # Add time/value to buffers
                current_time = time.time() - self.start_time
                self.time_buffer.append(current_time)
                self.value_buffer.append(selected_value)

                # Update plot
                self.plot_curve.setData(list(self.time_buffer), list(self.value_buffer))

        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            print(f"Error receiving data: {e}")

# App entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec_())
