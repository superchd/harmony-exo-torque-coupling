import os
import csv
import socket
import struct
import sys
import time
from datetime import datetime
from glob import glob

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea,
    QComboBox, QPushButton, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt

# ====== Save base dir (absolute) ======
SAVE_BASE_DIR = "/Users/hyundae/Desktop/harmonic-shr/Harmonic"  # recordings 하위로 저장

# ====== UDP Setup ======
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # Non-blocking mode

# ====== Joint Info ======
joint_names = [
    "shoulder_elevation",
    "shoulder_protraction",
    "shoulder_abduction",
    "shoulder_rotation",
    "shoulder_flexion",
    "elbow_flexion",
    "wrist_pronation"
]
SIDES = ["Right", "Left"]  # 순서 유지! Right 블록 다음 Left 블록

# 28개 채널의 메타(표시/저장용 컬럼명과 인덱스 매핑) 생성
def build_channel_schema():
    """
    반환:
      columns: CSV 컬럼명 리스트 (사람 친화적)
      idx_kind: [(src_index, kind), ...] kind ∈ {'deg','nm'}
    UI/수신 순서: side (Right, Left) × joint × [Position, Torque]
    Position 인덱스는 짝수, Torque 인덱스는 홀수.
    """
    columns = []
    idx_kind = []
    idx = 0
    for side in SIDES:
        for joint in joint_names:
            # Position -> degree
            columns.append(f"{side}_{joint}_degree")
            idx_kind.append((idx, "deg"))
            idx += 1
            # Torque -> Nm
            columns.append(f"{side}_{joint}_torque_nm")
            idx_kind.append((idx, "nm"))
            idx += 1
    return columns, idx_kind

CSV_VALUE_COLUMNS, IDX_KIND = build_channel_schema()

class JointDataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Joint Angles & Torques  —  Space = Start/Stop")
        self.setFocusPolicy(Qt.StrongFocus)

        # ===== recording/session state =====
        self.is_recording = False
        self.record_buffer = []
        self.start_time = None
        self.trial_id = 1
        self.max_tasks = 3                 # ★ 3가지 태스크
        self.max_reps_per_task = 10        # ★ 각 태스크 10회
        self.task_rep_counts = {t: 0 for t in range(1, self.max_tasks + 1)}
        self.session_dir = self._create_session_dir()

        # ===== UI =====
        self.value_labels = []  # 28 labels in UI order
        outer_layout = QVBoxLayout()

        # top controls
        top = QHBoxLayout()
        top.addWidget(QLabel("Task:", self))
        self.task_combo = QComboBox(self)
        self.task_combo.addItems([str(i) for i in range(1, self.max_tasks + 1)])
        top.addWidget(self.task_combo)

        self.status_label = QLabel("Status: IDLE", self)
        top.addWidget(self.status_label)

        self.counter_label = QLabel(self._counter_text(), self)
        self.counter_label.setMinimumWidth(300)
        top.addWidget(self.counter_label)

        self.btn_toggle = QPushButton("Start (Space)", self)
        self.btn_toggle.clicked.connect(self.toggle_recording)
        top.addWidget(self.btn_toggle)

        self.btn_export = QPushButton("Export Summary", self)
        self.btn_export.setToolTip("현재 세션의 모든 trial CSV를 하나로 합쳐 summary_all.csv와 각 task별 합본을 생성")
        self.btn_export.clicked.connect(self.export_summary)
        top.addWidget(self.btn_export)

        top.addStretch(1)
        outer_layout.addLayout(top)

        # scroll with readouts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Joint</b>"))
        header.addWidget(QLabel("<b>Value</b>"))
        lay.addLayout(header)

        # add rows in the same order as data stream
        for side in SIDES:
            for joint in joint_names:
                # Position (degrees)
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{side} {joint} - Position (degrees)"))
                lbl_pos = QLabel("0.000")
                row.addWidget(lbl_pos)
                lay.addLayout(row)
                self.value_labels.append(lbl_pos)
                # Torque (Nm)
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{side} {joint} - Torque (Nm)"))
                lbl_torque = QLabel("0.000")
                row.addWidget(lbl_torque)
                lay.addLayout(row)
                self.value_labels.append(lbl_torque)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_udp_data)
        self.timer.start(5)  # ~200 Hz

    # ===== utils =====
    def _create_session_dir(self):
        base_root = os.path.realpath(os.path.expanduser(SAVE_BASE_DIR))
        base = os.path.join(base_root, "recordings")
        os.makedirs(base, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(base, f"session_{ts}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _counter_text(self):
        done = sum(self.task_rep_counts.values())
        total = self.max_tasks * self.max_reps_per_task
        per_task = " | ".join([f"T{t}:{self.task_rep_counts[t]}/{self.max_reps_per_task}" for t in range(1, self.max_tasks+1)])
        return f"Trials: {done}/{total}  ||  {per_task}  ||  Next Trial ID: {self.trial_id}"

    def _current_task(self):
        return int(self.task_combo.currentText())

    def _next_rep_for_task(self, task_id):
        return self.task_rep_counts[task_id] + 1

    # ===== input =====
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_recording()
        else:
            super().keyPressEvent(event)

    def toggle_recording(self):
        if not self.is_recording:
            task_id = self._current_task()
            rep_idx = self._next_rep_for_task(task_id)
            if rep_idx > self.max_reps_per_task:
                self.status_label.setText(f"Status: MAX REPS reached for Task {task_id}")
                return
            self.is_recording = True
            self.record_buffer = []
            self.start_time = time.time()
            self.status_label.setText(f"Status: RECORDING (Task {task_id}, Rep {rep_idx})")
            self.btn_toggle.setText("Stop (Space)")
        else:
            self.is_recording = False
            offset_time = time.time()
            task_id = self._current_task()
            rep_idx = self._next_rep_for_task(task_id)
            self._write_current_trial_csv(task_id, rep_idx, self.record_buffer, self.start_time, offset_time)
            self.task_rep_counts[task_id] += 1
            self.trial_id += 1
            self.status_label.setText("Status: IDLE")
            self.btn_toggle.setText("Start (Space)")
            self.counter_label.setText(self._counter_text())

            # 모든 reps를 끝냈다면 다음 task로 자동 전환
            if self.task_rep_counts[task_id] >= self.max_reps_per_task:
                if task_id < self.max_tasks:
                    self.task_combo.setCurrentText(str(task_id + 1))
                else:
                    # 마지막 태스크까지 완료
                    if sum(self.task_rep_counts.values()) == self.max_tasks * self.max_reps_per_task:
                        QMessageBox.information(self, "Session Complete", "모든 태스크/반복 기록 완료! Export Summary를 눌러 합본을 생성하세요.")

    # ===== CSV I/O =====
    def _trial_header(self):
        return ["trial_id", "task_id", "rep_idx", "t_rel", "t_abs"] + CSV_VALUE_COLUMNS

    def _write_current_trial_csv(self, task_id, rep_idx, rows, t_onset_abs, t_offset_abs):
        fname = f"task{task_id}_rep{rep_idx:02d}.csv"
        fpath = os.path.join(self.session_dir, fname)

        header = self._trial_header()

        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"# onset_abs={t_onset_abs:.6f}",
                             f"offset_abs={t_offset_abs:.6f}",
                             f"duration_s={(t_offset_abs - t_onset_abs):.6f}",
                             f"session_dir={os.path.basename(self.session_dir)}"]) 
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

        print(f"[SAVED] {fpath} | rows: {len(rows)}")

    def _list_trial_csvs(self):
        return sorted(glob(os.path.join(self.session_dir, "task*_rep*.csv")))

    def export_summary(self):
        files = self._list_trial_csvs()
        if not files:
            QMessageBox.warning(self, "No files", "아직 저장된 trial CSV가 없습니다.")
            return

        header = self._trial_header()
        # 1) 전체 합본
        all_path = os.path.join(self.session_dir, "summary_all.csv")
        n_rows_total = 0
        with open(all_path, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow([f"# merged_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"n_files={len(files)}"]) 
            w.writerow(header)
            for fp in files:
                with open(fp, "r", newline="") as fin:
                    r = csv.reader(fin)
                    for row in r:
                        if not row:
                            continue
                        if row[0].startswith("#"):
                            continue  # 메타 라인 스킵
                        if row[0] == "trial_id":
                            continue  # 헤더 라인 스킵
                        w.writerow(row)
                        n_rows_total += 1
        print(f"[EXPORTED] {all_path} | rows: {n_rows_total}")

        # 2) Task별 합본
        for task_id in range(1, self.max_tasks + 1):
            task_files = sorted(glob(os.path.join(self.session_dir, f"task{task_id}_rep*.csv")))
            if not task_files:
                continue
            outp = os.path.join(self.session_dir, f"task{task_id}_all.csv")
            n_task_rows = 0
            with open(outp, "w", newline="") as fout:
                w = csv.writer(fout)
                w.writerow([f"# merged_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"task_id={task_id}", f"n_files={len(task_files)}"]) 
                w.writerow(header)
                for fp in task_files:
                    with open(fp, "r", newline="") as fin:
                        r = csv.reader(fin)
                        for row in r:
                            if not row:
                                continue
                            if row[0].startswith("#"):
                                continue
                            if row[0] == "trial_id":
                                continue
                            w.writerow(row)
                            n_task_rows += 1
            print(f"[EXPORTED] {outp} | rows: {n_task_rows}")

        QMessageBox.information(self, "Export Done", "summary_all.csv 와 task별 task*_all.csv 생성 완료")

    # ===== UDP polling =====
    def receive_udp_data(self):
        try:
            data, _ = sock.recvfrom(4096)
            if len(data) == 28 * 8:
                values = struct.unpack('28d', data)  # 28 channels

                # UI 업데이트: 짝수(포지션)만 degree 변환, 홀수(토크)는 Nm 그대로
                for i in range(28):
                    if i % 2 == 0:  # Position
                        self.value_labels[i].setText(f"{(values[i]*57.2958):.3f}")
                    else:           # Torque
                        self.value_labels[i].setText(f"{values[i]:.3f}")

                # 기록
                if self.is_recording and self.start_time is not None:
                    t_abs = time.time()
                    t_rel = t_abs - self.start_time
                    task_id = self._current_task()
                    rep_idx = self._next_rep_for_task(task_id)

                    # CSV 한 줄: 메타 5개 + 28개(사람 친화적 순서/단위)
                    row_values = []
                    for src_idx, kind in IDX_KIND:
                        v = values[src_idx]
                        if kind == "deg":
                            row_values.append(v * 57.2958)  # degree
                        else:
                            row_values.append(v)  # Nm

                    row = [
                        self.trial_id, task_id, rep_idx,
                        f"{t_rel:.6f}", f"{t_abs:.6f}"
                    ] + row_values

                    self.record_buffer.append(row)

        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Error receiving data: {e}")

# ===== App entry =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JointDataViewer()
    viewer.resize(780, 900)
    viewer.show()
    sys.exit(app.exec_())