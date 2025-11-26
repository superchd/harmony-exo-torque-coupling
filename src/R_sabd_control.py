import socket, time, threading, numpy as np, struct, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# works with command_and_report.cpp

commands = {}
start_setpoint = 90.*np.pi/180.
delta_theta_deg = 25

# CHANGED: r5_set -> r2_set
df = pd.DataFrame(columns=['time','r0_theta','r1_theta','r2_theta',
                           'r3_theta','r4_theta','r5_theta','r6_theta','r2_set'])

# CHANGED: r5_set -> r2_set
cols = ['time','r0_theta','r1_theta','r2_theta',
        'r3_theta','r4_theta','r5_theta','r6_theta','r2_set']
dict_list = []
now = datetime.now()

def command_loop(sock, target_ip, target_port, start, end, delay):
    for i in range(start, end + 1):
        angle_setpoint = (90.+delta_theta_deg if i%2==0 else 90.-delta_theta_deg)*np.pi/180.
        command_send(sock, target_ip, target_port, angle_setpoint)
        print(f"loop {i}")
        time.sleep(delay)

def command_send(sock, target_ip, target_port, command_setpoint):
    # CHANGED: EF_ -> R2_  (펌웨어가 인덱스 명령이면 아래 주석 라인으로 교체)
    message = f"R2_{command_setpoint:.6f}".encode('utf-8')
    # 인덱스 방식 사용 시: message = f"CMD 2 {command_setpoint:.6f}".encode('utf-8')

    send_time = time.time()
    sock.sendto(message, (target_ip, target_port))

    if dict_list:
        angles = list(map(dict_list[-1].get, cols[1:-1]))  # r0..r6
    else:
        angles = [np.nan]*7  # FIXED: 6 -> 7
    values = [send_time] + angles + [command_setpoint]
    dict_data = {col: val for col, val in zip(cols, values)}
    dict_list.append(dict_data)

def udp_receiver(sock, start, end, delay, timeout=5):
    sock.settimeout(timeout)
    duration = (end-start)*delay + 15
    start_time = time.time()
    while (time.time() - start_time) < duration:
        try:
            data, addr = sock.recvfrom(1024)
            recv_time = time.time()

            # (권장) 엔디안 명시: '<7d' (리틀엔디안일 때)
            angle_values = list(struct.unpack('<7d', data))  # 필요 시 '7d' 유지

            angle_setpoint = dict_list[-1][cols[-1]] if dict_list else start_setpoint
            values = [recv_time] + angle_values + [angle_setpoint]
            dict_data = {col: val for col, val in zip(cols, values)}
            dict_list.append(dict_data)

        except socket.timeout:
            print("Receiver timed out waiting for data, exiting.")
            break

def plot_data(filename, plot_vlines=True):
    df = pd.read_csv(filename)
    edges_idx = find_edges(df.r2_set)

    # CHANGED: r5_theta -> r2_theta, -1*set 제거
    plt.plot(df.time, df.r2_theta, label='measured angle (r2)')
    plt.plot(df.time, df.r2_set,   label='setpoint (r2)')

    deltat = 0.04
    if plot_vlines:
        for idx, edge_idx in enumerate(edges_idx):
            if idx == 0:
                plt.axvline(df.iloc[edge_idx]['time'] + deltat,  color='r', label="+0.04 s")
                plt.axvline(df.iloc[edge_idx]['time'] + 0.165,   color='k', label="+0.165 s")
            else:
                plt.axvline(df.iloc[edge_idx]['time'] + deltat,  color='r')
                plt.axvline(df.iloc[edge_idx]['time'] + 0.165,   color='k')

    plt.legend(); plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)')
    plt.title('R2 (Shoulder Abduction) Tracking')
    plt.show()

def find_edges(setpoints):
    edges_idx = []
    for i in range(1, len(setpoints)):
        if setpoints[i-1] != setpoints[i]:
            edges_idx.append(i)
    return edges_idx

if __name__ == "__main__":
    ONLY_PLOT = False
    file_only_plot = ""

    if not ONLY_PLOT:
        target_ip = "192.168.2.1"
        target_port = 12345

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 12346))

        start = 0; end = 10; delay = 5

        receiver_thread = threading.Thread(target=udp_receiver, args=(sock, start, end, delay,))
        receiver_thread.daemon = True
        receiver_thread.start()

        time.sleep(1)
        command_send(sock, target_ip, target_port, start_setpoint)
        time.sleep(5)

        command_loop(sock, target_ip, target_port, start, end, delay)

        command_send(sock, target_ip, target_port, start_setpoint)

        receiver_thread.join()
        sock.close()
        print("Done")

        df_final = pd.DataFrame.from_dict(dict_list)
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        filename = 'command-angle-R2-' + date_time + '.csv'
        df_final.to_csv(filename, index=False)
        plot_data(filename)
    else:
        plot_data(file_only_plot, plot_vlines=False)
