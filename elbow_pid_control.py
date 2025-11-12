#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import threading
import struct
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

###############################################################################
# 설정(필수 import 아래에 둘 것)
###############################################################################
# --- 기록/플롯 공통 ---
cols = ['time', 'r0_theta', 'r1_theta', 'r2_theta', 'r3_theta', 'r4_theta', 'r5_theta', 'r6_theta', 'r5_set']
dict_list = []
now = datetime.now()

# --- 네트워크 ---
TARGET_IP   = "192.168.2.1"  # Harmony PC IP
TARGET_PORT = 12345          # Harmony 수신 포트
LOCAL_PORT  = 12346          # 이 스크립트 수신 포트

# --- 과제(step) 모드 파라미터 ---
delta_theta_deg = 25.0
start_setpoint  = np.deg2rad(90.0)

# --- 제어 모드 ---
RESISTIVE_MODE  = True      # True면 제어 스레드 실행(아래 PD), False면 기존 step 과제 실행
CTRL_HZ         = 100       # 제어 주파수(50~200 권장)

# --- PD 제어(가상 임피던스) 파라미터 ---
# 목표: 평형각(theta_eq)에서 벗어나는 움직임에 "부드러운 저항감" 제공
THETA_EQ_DEG    = 90.0      # 평형각(deg)
KP              = 0.15      # P 이득(스프링 강성에 해당) [rad/rad] -> setpoint 오프셋 배율
KD              = 0.06      # D 이득(점성에 해당) [rad/(rad/s)] -> 속도에 비례한 오프셋
DERIV_LP_ALPHA  = 0.2       # 속도 추정 저역통과 계수(0~1, 클수록 빠름)
DEADBAND_VEL    = 0.02      # 작은 속도 무시 [rad/s]

# --- 지령(출력) 스무딩/제한 ---
CMD_LP_ALPHA    = 0.2       # 지령 저역통과 계수(0~1) — setpoint 자체도 부드럽게
MAX_CMD_RATE    = np.deg2rad(30)   # 지령 변화율 제한 [rad/s]
THETA_MIN       = np.deg2rad(45)   # 안전 하한
THETA_MAX       = np.deg2rad(135)  # 안전 상한

# --- 실행 시간 ---
RESISTIVE_RUN_SEC = 60      # 제어 실행 시간(초)

# --- 내부 상태 ---
stop_event = threading.Event()
r5_state = {
    "theta": None,
    "theta_prev": None,
    "vel_lp": 0.0,
    "t_recv": None,
    "t_recv_prev": None
}
theta_cmd_current = np.deg2rad(THETA_EQ_DEG)   # 실제 보낸 최신 지령
theta_cmd_filt    = np.deg2rad(THETA_EQ_DEG)   # 저역통과를 거친 지령(출력용)

###############################################################################
# 유틸 / 통신 함수
###############################################################################
def command_send(sock, target_ip, target_port, command_setpoint):
    """EF_{theta} 포맷으로 위치 지령 전송 + 최신 상태 로깅"""
    global dict_list

    message = f"EF_{command_setpoint:.6f}".encode('utf-8')
    send_time = time.time()
    sock.sendto(message, (target_ip, target_port))

    # 로그용: 마지막 수신 각도를 붙여 기록(없으면 NaN)
    if dict_list:
        angles = list(map(dict_list[-1].get, cols[1:-1]))  # r0..r6
    else:
        angles = [np.nan] * 7
    values = [send_time] + angles + [command_setpoint]
    dict_data = {c: v for c, v in zip(cols, values)}
    dict_list.append(dict_data)

def command_loop(sock, target_ip, target_port, start, end, delay):
    """±delta step 과제"""
    for i in range(start, end + 1):
        if i % 2 == 0:
            angle_setpoint = np.deg2rad(90.0 + delta_theta_deg)
        else:
            angle_setpoint = np.deg2rad(90.0 - delta_theta_deg)

        command_send(sock, target_ip, target_port, angle_setpoint)
        print(f"[STEP] loop {i}, cmd={angle_setpoint:.3f} rad")
        time.sleep(delay)

def udp_receiver(sock, start, end, delay, timeout=5):
    """Harmony에서 온 7개 double(r0..r6)을 수신하고 로깅 + r5 상태 업데이트"""
    global dict_list, r5_state

    sock.settimeout(timeout)
    duration = (end - start) * delay + 15 if not RESISTIVE_MODE else (RESISTIVE_RUN_SEC + 5)
    start_time = time.time()

    while (time.time() - start_time) < duration and not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
            recv_time = time.time()

            # 최소 7*8=56 bytes 필요
            if len(data) < 56:
                continue

            try:
                angle_values = list(struct.unpack('7d', data[:56]))
            except struct.error:
                continue

            # 현재 setpoint 추적(마지막 dict_list의 r5_set 사용)
            angle_setpoint = dict_list[-1][cols[-1]] if dict_list else start_setpoint

            # 로깅
            values = [recv_time] + angle_values + [angle_setpoint]
            dict_data = {c: v for c, v in zip(cols, values)}
            dict_list.append(dict_data)

            # r5 상태 업데이트
            r5_state["theta_prev"]   = r5_state["theta"]
            r5_state["t_recv_prev"]  = r5_state["t_recv"]
            r5_state["theta"]        = angle_values[5]  # r5_theta
            r5_state["t_recv"]       = recv_time

        except socket.timeout:
            print("[RX] timeout, exit receiver")
            break

###############################################################################
# PD 컨트롤러 (I 없음)
###############################################################################
def pd_resistive_controller(sock, target_ip, target_port):
    """
    '평형각(theta_eq)' 기준의 PD(스프링+댐퍼)로 setpoint를 미세 조정하여 저항감 제공.
    - D는 측정치 기반(vel)으로 계산 => step 변화 시 D-kick 방지
    - setpoint 자체도 1차 LPF + 변화율 제한 + 안전 클램프
    """
    global theta_cmd_current, theta_cmd_filt

    Ts = 1.0 / CTRL_HZ
    theta_eq = np.deg2rad(THETA_EQ_DEG)

    next_t = time.monotonic()

    while not stop_event.is_set():
        # 주기 정렬 (드리프트 최소화)
        now_t = time.monotonic()
        if now_t < next_t:
            time.sleep(next_t - now_t)
        next_t += Ts
        t0 = time.time()

        th          = r5_state["theta"]
        th_prev     = r5_state["theta_prev"]
        t_recv      = r5_state["t_recv"]
        t_recv_prev = r5_state["t_recv_prev"]

        if th is None or th_prev is None or t_recv is None or t_recv_prev is None:
            continue

        # 실제 수신 간격 기반 속도 추정
        dt_meas = max(1e-3, t_recv - t_recv_prev)
        vel_raw = (th - th_prev) / dt_meas

        # 소속도 무시(미세 떨림 제거) + 1차 LPF
        if abs(vel_raw) < DEADBAND_VEL:
            vel_raw = 0.0
        r5_state["vel_lp"] = (1 - DERIV_LP_ALPHA) * r5_state["vel_lp"] + DERIV_LP_ALPHA * vel_raw
        vel = r5_state["vel_lp"]  # 미분(속도) 추정치

        # PD 제어: setpoint 오프셋 = -Kp*(th - theta_eq) - Kd*vel
        #  - 위치가 평형각보다 +방향이면 setpoint를 -쪽으로 당겨 '복원력' 생성
        #  - +방향으로 빠르게 움직이면 setpoint를 더 -쪽으로 당겨 '점성 저항' 생성
        pos_err   = (th - theta_eq)          # 측정치 기준 오차(참조 = theta_eq)
        offset_pd = -KP * pos_err - KD * vel

        theta_target = theta_eq + offset_pd

        # 안전 각도 제한
        theta_target = float(np.clip(theta_target, THETA_MIN, THETA_MAX))

        # 지령 저역통과(출력 스무딩) -> setpoint 자체를 더 매끈하게
        theta_cmd_filt = (1 - CMD_LP_ALPHA) * theta_cmd_filt + CMD_LP_ALPHA * theta_target

        # 지령 변화율 제한(사용자/기계 안전)
        max_step = MAX_CMD_RATE * Ts
        step = float(np.clip(theta_cmd_filt - theta_cmd_current, -max_step, max_step))
        theta_cmd_current += step

        # 전송
        command_send(sock, target_ip, target_port, theta_cmd_current)

        # (옵션) 남은 시간 슬립으로 주기 유지 (위의 주기 정렬로 대체 가능)
        dt_sleep = Ts - (time.time() - t0)
        if dt_sleep > 0:
            time.sleep(dt_sleep)

###############################################################################
# 플로팅/후처리
###############################################################################
def find_edges(setpoints):
    edges_idx = []
    for i in range(1, len(setpoints)):
        if setpoints[i - 1] != setpoints[i]:
            edges_idx.append(i)
    return edges_idx

def plot_data(filename, plot_vlines=True):
    df = pd.read_csv(filename)
    edges_idx = find_edges(df.r5_set)

    plt.plot(df.time, df.r5_theta, label='measured angle')
    plt.plot(df.time, df.r5_set, label='setpoint')

    deltat = 0.04
    if plot_vlines and len(edges_idx) > 0:
        for idx, edge_idx in enumerate(edges_idx):
            if idx == 0:
                plt.axvline(df.iloc[edge_idx]['time'] + deltat, color='r', label="0.04 s after command")
                plt.axvline(df.iloc[edge_idx]['time'] + 0.165, color='k', label="0.165 s after command")
            else:
                plt.axvline(df.iloc[edge_idx]['time'] + deltat, color='r')
                plt.axvline(df.iloc[edge_idx]['time'] + 0.165, color='k')

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(filename)
    plt.show()

###############################################################################
# 메인
###############################################################################
if __name__ == "__main__":
    # 과거 CSV만 그림 그리고 싶으면 True
    ONLY_PLOT = False
    file_only_plot = ""  # 예: 'command-angle-11-11-2025_15-10-00.csv'

    if ONLY_PLOT:
        plot_data(file_only_plot, plot_vlines=False)
        raise SystemExit(0)

    # 소켓 준비
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", LOCAL_PORT))

    # 수신 스레드 시작
    start = 0
    end   = 10
    delay = 5

    receiver_thread = threading.Thread(target=udp_receiver, args=(sock, start, end, delay))
    receiver_thread.daemon = True
    receiver_thread.start()

    # 초기 평형각으로 세팅
    time.sleep(0.5)
    theta_cmd_current = np.deg2rad(THETA_EQ_DEG)
    theta_cmd_filt    = theta_cmd_current
    command_send(sock, TARGET_IP, TARGET_PORT, theta_cmd_current)
    time.sleep(0.5)

    if RESISTIVE_MODE:
        # PD 저항 모드: 고주기 제어 스레드
        ctrl_thread = threading.Thread(target=pd_resistive_controller, args=(sock, TARGET_IP, TARGET_PORT))
        ctrl_thread.daemon = True
        ctrl_thread.start()

        # 설정 시간 동안 실행
        time.sleep(RESISTIVE_RUN_SEC)
        stop_event.set()
        ctrl_thread.join()
    else:
        # 기존 step 과제
        command_loop(sock, TARGET_IP, TARGET_PORT, start, end, delay)
        command_send(sock, TARGET_IP, TARGET_PORT, start_setpoint)

    # 수신 스레드 종료 대기
    receiver_thread.join()

    # 소켓 닫기
    sock.close()
    print("Done")

    # CSV 저장 + 플롯
    df_final = pd.DataFrame.from_dict(dict_list)
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    filename = 'command-angle-' + date_time + '.csv'
    df_final.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    plot_data(filename)
