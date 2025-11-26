import socket, time, struct, numpy as np

TARGET_IP="192.168.2.1"; TARGET_PORT=12345
LOCAL_PORT=12346
UNPACK_ENDIAN="<"
STEP_DEG=8.0     # 좀 더 확실히 보이게 약간 키움
HOLD=0.7

# 펌웨어가 쓸 법한 태그 후보들 (필요하면 추가/삭제)
TAG_CANDIDATES = [
    "EF_",        # Elbow flexion (지금 유일하게 먹는 태그)
    "SABD_",      # Shoulder abduction (임상 DOF)
    "SPRO_",      # Shoulder protraction
    "SROT_",      # Shoulder rotation
    "SFLEX_",     # Shoulder flexion
    "SELEV_",     # Shoulder elevation
    "WPRO_",      # Wrist pronation
    "R0_", "R1_", "R2_", "R3_", "R4_", "R5_", "R6_",  # 혹시 모를 R#_류
]

def send_tag(sock, tag, rad):
    msg = f"{tag}{rad:.6f}".encode("utf-8")
    sock.sendto(msg, (TARGET_IP, TARGET_PORT))

def recv_many(sock, dur=0.8):
    t0=time.time(); buf=[]
    sock.settimeout(0.2)
    while time.time()-t0<dur:
        try:
            data,_=sock.recvfrom(1024)
            if len(data)==7*8:
                vals=struct.unpack(f"{UNPACK_ENDIAN}7d", data)
                buf.append(vals)
        except Exception:
            pass
    return np.array(buf) if buf else None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LOCAL_PORT))

base = np.deg2rad(90)
time.sleep(0.5)

for tag in TAG_CANDIDATES:
    # +step
    send_tag(sock, tag, base + np.deg2rad(STEP_DEG)); time.sleep(HOLD)
    A = recv_many(sock)
    # -step
    send_tag(sock, tag, base - np.deg2rad(STEP_DEG)); time.sleep(HOLD)
    B = recv_many(sock)
    # back
    send_tag(sock, tag, base); time.sleep(0.5)
    C = recv_many(sock)

    resp=-1.0; arg=None
    for mat in [A,B,C]:
        if mat is None: continue
        s=np.nanstd(mat, axis=0)     # r0..r6
        if s.max()>resp:
            resp=s.max(); arg=int(np.nanargmax(s))
    print(f"[{tag:<6}] -> most varying r{arg}, std={resp:.4f}")

sock.close()
