import socket
import time
import threading
import numpy as np
import struct
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# works with command_and_report.cpp

# Global variables to store event and data times
commands = {}
start_setpoint = 90.*np.pi/180.
delta_theta_deg = 25 #change from 5

df = pd.DataFrame(columns=['time', 'r0_theta', 'r1_theta', 'r2_theta', \
                           'r3_theta', 'r4_theta', 'r5_theta', 'r6_theta', 'r5_set'])

cols = ['time', 'r0_theta', 'r1_theta', 'r2_theta', \
                           'r3_theta', 'r4_theta', 'r5_theta', 'r6_theta', 'r5_set']
dict_list = []
now = datetime.now() # current date and time

def command_loop(sock, target_ip, target_port, start, end, delay):
    for i in range(start, end + 1):
        if i%2 == 0:
            angle_setpoint = (90.+delta_theta_deg)*np.pi/180.
        else:
            angle_setpoint = (90.-delta_theta_deg)*np.pi/180.


        command_send(sock, target_ip, target_port, angle_setpoint)
        print(f"loop {i}")
        time.sleep(delay)  # Wait 5 seconds

def command_send(sock, target_ip, target_port, command_setpoint):
    message = f"EF_{command_setpoint:.6f}".encode('utf-8')
    send_time = time.time()
    sock.sendto(message, (target_ip, target_port))
    # commands[send_time] = angle_setpoint

    if dict_list:
        angles = list( map(dict_list[-1].get, cols[1:-1]) )
    else:
        angles = [np.nan]*6
    values = [send_time] + angles + [command_setpoint]
    dict_data = {col: value for col,value in zip(cols,values)}
    dict_list.append(dict_data)


def udp_receiver(sock, start, end, delay, timeout=5):
    sock.settimeout(timeout)  # Set timeout for the recvfrom call
    duration = (end-start)*delay + 15 # Run for 15 seconds more than end-start time
    start_time = time.time()
    while ((time.time() - start_time) < duration ):
        try:
            data, addr = sock.recvfrom(1024)
            recv_time = time.time()

            # Unpack the received data assuming it contains 7 doubles (64-bit each)
            # r0_theta, r1_theta, r2_theta, r3_theta, r4_theta, r5_theta, r6_theta = values
            angle_values = list(struct.unpack('7d', data))
            
            if dict_list:
                angle_setpoint = dict_list[-1][cols[-1]]
            else:
                angle_setpoint = start_setpoint

            values = [recv_time] + angle_values + [angle_setpoint]
            dict_data = {col: value for col,value in zip(cols,values)}
            dict_list.append(dict_data)

        except socket.timeout:
            print("Receiver timed out waiting for data, exiting.")
            break


def plot_data(filename, plot_vlines = True):

    df = pd.read_csv(filename)

    edges_idx = find_edges(df.r5_set)

    plt.plot(df.time, df.r5_theta, label='measured angle')
    plt.plot(df.time, -1*df.r5_set, label='setpoint')

    deltat = 0.04

    if plot_vlines:
        for idx, edge_idx in enumerate(edges_idx):
            if idx == 0:
                plt.axvline(df.iloc[edge_idx]['time'] + deltat,color='r', label="0.04 Seconds after command")
                plt.axvline(df.iloc[edge_idx]['time'] + 0.165,color='k', label="0.165 Seconds after command")
            else:
                plt.axvline(df.iloc[edge_idx]['time'] + deltat,color='r')
                plt.axvline(df.iloc[edge_idx]['time'] + 0.165,color='k')


    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.show()

def find_edges(setpoints):

    edges_idx = []
    for i in range(1, len(setpoints)):
        if setpoints[i-1] != setpoints[i]:
            edges_idx.append(i)

    return edges_idx



if __name__ == "__main__":

    # Set to true and give file name if you want to plot from a previously recorded set of data
    ONLY_PLOT = False
    file_only_plot = ""


    if not ONLY_PLOT:

        target_ip = "192.168.2.1"  # IP address of Harmony computer
        target_port = 12345       # Replace with the actual target port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 12346))  # Bind to a different port to listen for responses

        start = 0
        end = 10
        delay = 5 #changed from 5


        # Create a thread for the receiver
        receiver_thread = threading.Thread(target=udp_receiver, args=(sock,start,end,delay,))
        receiver_thread.daemon = True
        receiver_thread.start()

        # Set setpoint to start setpoint
        time.sleep(1)
        command_send(sock, target_ip, target_port, start_setpoint)
        time.sleep(5)


        command_loop(sock, target_ip, target_port, start, end, delay)

        command_send(sock, target_ip, target_port, start_setpoint)

        # Optionally, wait for the receiver thread to finish
        receiver_thread.join()

        

        # Close the socket after the receiver thread has finished
        sock.close()

        print("Done")


        df_final = pd.DataFrame.from_dict(dict_list)
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

        filename = 'command-angle-' + date_time + '.csv'
        df_final.to_csv(filename, index=False)

        plot_data(filename)
    
    else:

        plot_data(file_only_plot, plot_vlines=False)
