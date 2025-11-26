import socket
import time
import threading
import matplotlib.pyplot as plt

# Works with udp_echo_server.cpp

# Global variable to store the round-trip time
round_trip_times = {}

def udp_sender(sock, target_ip, target_port, start=1, end=100):
    for i in range(start, end + 1):
        message = f"{i}_{time.time()}".encode('utf-8')
        sock.sendto(message, (target_ip, target_port))
        print(f"Sent: {i}")
        time.sleep(0.1)  # Simulate some delay (optional)
    # Signal the receiver thread to stop by sending a special message
    sock.sendto(b"STOP", (target_ip, target_port))


def udp_receiver(sock, timeout=5):
    sock.settimeout(timeout)  # Set timeout for the recvfrom call
    while True:
        data, addr = sock.recvfrom(1024)

        if data == b"STOP":
            print("Received stop signal, exiting receiver.")
            break
        # try:
        # except:
        #     raise('socket closed')
        received_number, send_time = data.decode('utf-8').split('_')
        received_number = int(received_number)
        send_time = float(send_time)
        round_trip_time = time.time() - send_time
        if (received_number not in round_trip_times):
            round_trip_times[received_number] = round_trip_time
            print(f"Received: {received_number} | Round-Trip Latency: {round_trip_time:.6f} seconds")

if __name__ == "__main__":
    target_ip = "192.168.2.1"  # Replace with the actual target IP
    target_port = 12345       # Replace with the actual target port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 12346))  # Bind to a different port to listen for responses

    # Create a thread for the receiver
    receiver_thread = threading.Thread(target=udp_receiver, args=(sock,))
    receiver_thread.daemon = True
    receiver_thread.start()

    # Run the sender function in the main thread
    udp_sender(sock, target_ip, target_port, end=50)

    
    # Optionally, wait for the receiver thread to finish
    receiver_thread.join()

    # Close the socket after the receiver thread has finished
    sock.close()

    # Convert the round-trip times dictionary to a list
    round_trip_time_values = list(round_trip_times.values())

    # Plot the histogram of round-trip times
    plt.hist(round_trip_time_values, bins=10, edgecolor='black')
    plt.title('Histogram of Round-Trip Times')
    plt.xlabel('Round-Trip Time (seconds)')
    plt.ylabel('Frequency')
    plt.show()