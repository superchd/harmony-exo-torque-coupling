import socket, time, struct, numpy as np

TARGET_IP="192.168.2.1"; TARGET_PORT=12345
LOCAL_PORT=12346
UNPACK_ENDIAN="<"   # 리틀엔디안 가정
STEP_DEG=5.0
HOLD=0.7

# 서버가 이해하는 태그 맵 (필요 시 여기만 수정)
TAG_MAP = {
    0: "R0_",
    1: "R1_",
    2: "R2_",   # shoulder_abduction 가정
    3: "R3_",
    4: "R4_",
    5: "EF_",   # elbow flexion은 보통 EF_로만 받는 펌웨어가 많음
    6: "R6_",
}

def send_cmd(sock, jidx, rad):
    tag = TAG_MAP[jidx]
    # 언더스코어 형식: 예) "R2_1.570796" / "EF_0.785398"
    msg = f"{tag}{rad:.6f}".encode("utf-8")
    sock.sendto(msg, (TARGET_IP, TARGET_PORT))

def recv_many(sock, dur=0.8):
    t0 = time.time(); buf=[]
    sock.settimeout(0.2)
    while time.time()-t0 < dur:
        try:
            data,_ = sock.recvfrom(1024)
            if len(data) == 7*8:
                vals = struct.unpack(f"{UNPACK_ENDIAN}7d", data)
                buf.append(vals)
        except Exception:
            pass
    return np.array(buf) if buf else None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LOCAL_PORT))

base = np.deg2rad(90)
time.sleep(0.5)

for j in range(7):
    # +step
    send_cmd(sock, j, base + np.deg2rad(STEP_DEG)); time.sleep(HOLD)
    A = recv_many(sock)

    # -step
    send_cmd(sock, j, base - np.deg2rad(STEP_DEG)); time.sleep(HOLD)
    B = recv_many(sock)

    # back to base
    send_cmd(sock, j, base); time.sleep(0.5)
    C = recv_many(sock)

    # 반응량(표준편차 최대값)으로 어느 r#가 가장 변했는지 평가
    resp = -1.0; arg = None
    for mat in [A,B,C]:
        if mat is None: continue
        s = np.nanstd(mat, axis=0)  # r0..r6
        if s.max() > resp:
            resp = s.max(); arg = int(np.nanargmax(s))
    print(f"probe tag {TAG_MAP[j]:<4} -> most varying r{arg}, std={resp:.4f}")

sock.close()
