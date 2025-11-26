import socket
import time
import threading
import numpy as np
import struct
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

"""
Harmony command-and-report용 범용 조인트 제어 스크립트
- JOINT_IDX로 제어할 조인트를 선택
- TAG_BY_IDX에서 펌웨어가 기대하는 태그 문자열 확인/수정
- 수신 패킷은 r0~r6 순서로 7개의 double이라 가정
"""

# ==========================
# 사용자 설정
# ==========================
JOINT_IDX = 5  # 0~6 중 선택, 예) 5 = Elbow Flexion
TAG_BY_IDX = {0: "R0_", 1: "R1_", 2: "R2_", 3: "R3_", 4: "R4_", 5: "EF_", 6: "R6_"}  # 펌웨어 태그 확인 필요

target_ip = "192.168.2.1"   # Harmony 컴퓨터 IP
target_port = 12345         # Harmony 수신 포트
local_port = 12346          # 로컬 수신 포트

start_setpoint = np.deg2rad(90.0)
delta_theta_deg = 25
start_loop = 0
end_loop = 10
loop_delay_sec = 5

# 플롯에서 setpoint 부호를 뒤집어야 할 때 True로 변경
PLOT_NEGATE_SETPOINT = False

# 과거 CSV만 플롯하고 싶으면 True로 바꾸고 경로 지정
ONLY_PLOT = False
file_only_plot = ""  # 예: "command-angle-09-24-2025_11-20-30.csv"

# ==========================
# 내부 전역
# ==========================
theta_cols = [f"r{i}_theta" for i in range(7)]
SET_COL = f"r{JOINT_IDX}_set"
cols = ["time"] + theta_cols + [SET_COL]
dict_list = []


def command_send(sock, target_ip, target_port, command_setpoint):
    """선택된 조인트에 setpoint를 송신하고 직전 각도와 함께 로그에 기록"""
    tag = TAG_BY_IDX[JOINT_IDX]
    message = f"{tag}{command_setpoint:.6f}".encode('utf-8')
    send_time = time.time()
    sock.sendto(message, (target_ip, target_port))

    # 직전 수신 각도를 함께 기록해두기
    if dict_list:
        angles = list(map(dict_list[-1].get, theta_cols))
    else:
        angles = [np.nan] * 7

    values = [send_time] + angles + [command_setpoint]
    dict_data = {col: val for col, val in zip(cols, values)}
    dict_list.append(dict_data)


def command_loop(sock, target_ip, target_port, start, end, delay):
    """짝수/홀수 루프마다 setpoint를 +/− delta로 토글 전송"""
    for i in range(start, end + 1):
        if i % 2 == 0:
            angle_setpoint = np.deg2rad(90.0 + delta_theta_deg)
        else:
            angle_setpoint = np.deg2rad(90.0 - delta_theta_deg)
        command_send(sock, target_ip, target_port, angle_setpoint)
        print(f"loop {i}")
        time.sleep(delay)


def udp_receiver(sock, start, end, delay, timeout=5):
    """UDP로 r0~r6 각도 7개(double)를 수신하고 현재 setpoint와 함께 로그"""
    sock.settimeout(timeout)
    duration = (end - start) * delay + 15  # 루프 시간보다 조금 더 길게
    start_time = time.time()
    while (time.time() - start_time) < duration:
        try:
            data, addr = sock.recvfrom(1024)
            recv_time = time.time()

            # r0_theta, r1_theta, ..., r6_theta 순서의 7개 double
            angle_values = list(struct.unpack('7d', data))

            if dict_list:
                angle_setpoint = dict_list[-1].get(SET_COL, start_setpoint)
            else:
                angle_setpoint = start_setpoint

            values = [recv_time] + angle_values + [angle_setpoint]
            dict_data = {col: val for col, val in zip(cols, values)}
            dict_list.append(dict_data)

        except socket.timeout:
            print("Receiver timed out waiting for data, exiting.")
            break


def find_edges(setpoints):
    """연속 원소가 변하는 지점의 인덱스를 반환"""
    edges_idx = []
    for i in range(1, len(setpoints)):
        if setpoints[i - 1] != setpoints[i]:
            edges_idx.append(i)
    return edges_idx


def plot_data(filename, joint_idx=None, plot_vlines=True, negate_set=False):
    """CSV 파일에서 선택 조인트의 measured vs setpoint를 플롯"""
    if joint_idx is None:
        joint_idx = JOINT_IDX

    df = pd.read_csv(filename)
    theta_col = f"r{joint_idx}_theta"
    set_col = f"r{joint_idx}_set"

    if negate_set:
        set_series = -1.0 * df[set_col].values
    else:
        set_series = df[set_col].values

    edges_idx = find_edges(set_series)

    plt.plot(df.time, df[theta_col], label='measured angle')
    plt.plot(df.time, set_series, label='setpoint')

    deltat = 0.04
    if plot_vlines and len(edges_idx) > 0:
        for k, ei in enumerate(edges_idx):
            t0 = df.iloc[ei]['time']
            if k == 0:
                plt.axvline(t0 + deltat, color='r', label="0.04 s after command")
                plt.axvline(t0 + 0.165, color='k', label="0.165 s after command")
            else:
                plt.axvline(t0 + deltat, color='r')
                plt.axvline(t0 + 0.165, color='k')

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f"Joint r{joint_idx} response")
    plt.show()


if __name__ == "__main__":

    if ONLY_PLOT:
        plot_data(file_only_plot, joint_idx=JOINT_IDX, plot_vlines=False, negate_set=PLOT_NEGATE_SETPOINT)
    else:
        # 소켓 준비
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", local_port))

        # 수신 스레드 시작
        receiver_thread = threading.Thread(
            target=udp_receiver,
            args=(sock, start_loop, end_loop, loop_delay_sec,)
        )
        receiver_thread.daemon = True
        receiver_thread.start()

        # 시작 setpoint로 정렬
        time.sleep(1)
        command_send(sock, target_ip, target_port, start_setpoint)
        time.sleep(5)

        # 토글 명령 루프
        command_loop(sock, target_ip, target_port, start_loop, end_loop, loop_delay_sec)

        # 종료 전 기준자세로 복귀
        command_send(sock, target_ip, target_port, start_setpoint)

        # 수신 스레드 종료 대기
        receiver_thread.join()

        # 소켓 닫기
        sock.close()
        print("Done")

        # 로그 저장 및 플롯
        df_final = pd.DataFrame.from_dict(dict_list)
        date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        filename = f'command-angle-r{JOINT_IDX}-{date_time}.csv'
        df_final.to_csv(filename, index=False)

        plot_data(filename, joint_idx=JOINT_IDX, plot_vlines=True, negate_set=PLOT_NEGATE_SETPOINT)
