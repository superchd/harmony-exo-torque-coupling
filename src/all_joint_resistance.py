import socket
import struct
import time
import threading

import numpy as np

# ====== 네트워크 설정 ======
HARMONY_IP   = "192.168.2.1"   # Harmony SHR PC IP
HARMONY_PORT = 12345           # C++ command_and_report가 listen 하는 포트
LOCAL_PORT   = 12346           # 이 파이썬이 데이터를 받는 포트 (C++에서 TARGET_PORT로 쓰는 값)

N_JOINTS = 7  # 오른팔 7개 조인트

# 제어할 조인트 선택 (1~7 중 선택)
CONTROL_JOINT_IDS = [1, 2, 3, 4, 5, 6, 7]
# 예: 어깨 외전(J3) + 팔꿈치(J5)만 제어하고 싶으면 [3, 5] 로 바꾸면 됨.

# ====== PD 게인 (조인트별로 다르게 줄 수도 있음) ======
Kp_default = 8.0   # 위치 에러에 대한 비례 이득
Kd_default = 0.5   # 속도에 대한 미분 이득

# 조인트별 Kp, Kd 딕셔너리 (원하면 조인트마다 따로 조절 가능)
Kp = {jid: Kp_default for jid in CONTROL_JOINT_IDS}
Kd = {jid: Kd_default for jid in CONTROL_JOINT_IDS}

# ====== 상태 변수 ======
joint_states = {
    j: {
        "theta": 0.0,        # 현재 각도 [rad]
        "theta_ref": None,   # 참조(고정하고 싶은) 각도 [rad]
        "vel": 0.0,          # 저역통과 속도 [rad/s]
        "t_prev": None,      # 이전 샘플 시간
        "theta_prev": None,  # 이전 각도
    }
    for j in range(1, N_JOINTS + 1)
}

lock = threading.Lock()  # 수신 스레드/제어 루프 동기화용


# ====== UDP 수신 스레드 ======
def udp_receiver(sock, running_flag):
    """
    C++에서 보내는 7개 조인트 각도(double)를 계속 받아서 joint_states에 저장.
    """
    sock.settimeout(1.0)
    print("[Receiver] 시작")

    alpha = 0.2  # 속도 저역통과 필터 계수

    while running_flag["run"]:
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            continue

        if len(data) < 8 * N_JOINTS:
            # 데이터 길이가 이상하면 무시
            continue

        vals = struct.unpack("7d", data[:8 * N_JOINTS])
        t_now = time.time()

        with lock:
            for idx in range(N_JOINTS):
                jid = idx + 1  # 배열 index(0~6) → 조인트 번호(1~7)
                st = joint_states[jid]

                theta = vals[idx]
                if st["t_prev"] is not None:
                    dt = t_now - st["t_prev"]
                    if dt > 0:
                        vel_raw = (theta - st["theta_prev"]) / dt
                        # 저역통과 필터
                        st["vel"] = (1.0 - alpha) * st["vel"] + alpha * vel_raw

                st["theta"] = theta
                st["t_prev"] = t_now
                st["theta_prev"] = theta


# ====== 제어 루프 ======
def control_loop(sock, running_flag):
    """
    1) 몇 초 동안 현재 자세를 읽고 참조각(theta_ref)로 저장
    2) 이후에는 PD 제어로 각 조인트를 그 각도 주변에 붙잡아 둠.
    """
    # 1) 초기 자세를 참조로 잡기 위해 잠깐 대기
    print("[Control] 초기 자세 읽는 중... (2초)")
    time.sleep(2.0)

    with lock:
        for jid in CONTROL_JOINT_IDS:
            st = joint_states[jid]
            st["theta_ref"] = st["theta"]
            print(f"[Control] Joint {jid} reference angle = {st['theta_ref']:.3f} rad")

    print("[Control] 참조각 고정, PD 제어 시작")

    dt = 0.01  # 제어 주기 10 ms (100 Hz)

    while running_flag["run"]:
        t0 = time.time()

        with lock:
            for jid in CONTROL_JOINT_IDS:
                st = joint_states[jid]
                if st["theta_ref"] is None:
                    continue  # 아직 참조각을 못 잡았으면 스킵

                theta = st["theta"]
                theta_ref = st["theta_ref"]
                vel = st["vel"]

                # 에러 정의: ref - actual  (ref에서 벗어나면 복원하려고 함)
                e = theta_ref - theta
                de = -vel  # vel ≈ d(theta)/dt 이므로, de ≈ d(e)/dt = -vel

                u = Kp[jid] * e + Kd[jid] * de

                # C++에서 "J{조인트번호}_{값}" 형식으로 파싱한다고 가정
                msg = f"J{jid}_{u:.6f}".encode("utf-8")
                sock.sendto(msg, (HARMONY_IP, HARMONY_PORT))

        # 주기 맞추기
        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    # UDP 소켓 생성 및 바인딩
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", LOCAL_PORT))
    print(f"[Main] UDP bound on port {LOCAL_PORT}")

    running_flag = {"run": True}

    # 수신 스레드 시작
    recv_thread = threading.Thread(target=udp_receiver, args=(sock, running_flag))
    recv_thread.daemon = True
    recv_thread.start()

    try:
        control_loop(sock, running_flag)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C 감지, 종료 중...")
    finally:
        running_flag["run"] = False
        recv_thread.join()
        sock.close()
        print("[Main] 종료 완료")


if __name__ == "__main__":
    main()
