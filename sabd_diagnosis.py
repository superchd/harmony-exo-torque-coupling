#!/usr/bin/env python3
import socket, struct, time
UDP_IP, UDP_PORT = "0.0.0.0", 12345
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((UDP_IP, UDP_PORT)); s.setblocking(False)

mins = [float("inf")]*7  # right angles 0..6
maxs = [float("-inf")]*7
t0 = time.time()
print(">> 외전만 천천히 5초간 수행하세요...")
while time.time() - t0 < 5.0:
    try:
        data,_ = s.recvfrom(1024)
    except BlockingIOError:
        continue
    if len(data) != 28*8: 
        continue
    vals = struct.unpack('28d', data)
    for i in range(7): # right angles 0..6
        v = vals[i]
        mins[i] = v if v < mins[i] else mins[i]
        maxs[i] = v if v > maxs[i] else maxs[i]

ranges = [(i, maxs[i]-mins[i], mins[i], maxs[i]) for i in range(7)]
ranges.sort(key=lambda x: x[1], reverse=True)
print("\n== 변화폭 Ranking (Right angles 0..6) ==")
for i, span, mn, mx in ranges:
    print(f"idx {i}: Δ={span:.4f}  min={mn:.4f}  max={mx:.4f}")
print("\n보통 외전이면 idx 2가 1위여야 정상입니다. (단위가 deg이면 Δ가 훨씬 크게 보입니다)")
