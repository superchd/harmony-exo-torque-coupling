import socket
import struct
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame
)
from PyQt5.QtCore import QTimer

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

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Joint Angles and Torques")
        self.value_labels = []

        # Layout setup
        outer_layout = QVBoxLayout()
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
        self.setLayout(outer_layout)

        # Timer for polling UDP data
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # 5 ms for 200 Hz polling

    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) == 28 * 8:
                values = struct.unpack('28d', data)
                for i in range(28):
                    self.value_labels[i].setText(f"{values[i]:.3f}")
        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            print(f"Error receiving data: {e}")

# App entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.show()
    sys.exit(app.exec_())
