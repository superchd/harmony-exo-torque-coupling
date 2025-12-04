import socket
import time
import math

# Harmony 쪽 IP / 포트
HARMONY_IP = "192.168.2.1"  # Harmony PC IP (네 환경에 맞게 조정)
HARMONY_PORT = 12345        # command_and_report.cpp 가 bind 한 포트

# Shoulder Abduction 각도 설정 (라디안)
SA_START = 0.0                       # 기준 자세(0 rad)
SA_DEG = 20.0                        # 20도 정도 외전
SA_TARGET = SA_DEG * math.pi / 180.  # deg -> rad

# 반복 전송 주기
DT = 0.02        # 20 ms (50 Hz)
DURATION = 5.0   # 5초 동안 유지

def send_sa(sock, value_rad):
    """SA(Shoulder Abduction)에 명령 보내기"""
    msg = f"SA_{value_rad:.6f}".encode("utf-8")
    sock.sendto(msg, (HARMONY_IP, HARMONY_PORT))

if __name__ == "__main__":
    # UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("SA baseline(0 rad)으로 맞추는 중...")
    send_sa(sock, SA_START)
    time.sleep(1.0)

    print(f"SA {SA_DEG}도(≈{SA_TARGET:.3f} rad)로 5초 동안 유지합니다...")
    start_time = time.time()
    while time.time() - start_time < DURATION:
        send_sa(sock, SA_TARGET)
        time.sleep(DT)

    print("SA 다시 baseline(0 rad)으로 복귀...")
    send_sa(sock, SA_START)
    time.sleep(1.0)

    sock.close()
    print("완료.")
