#include "research_interface.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <thread>

/**
 * @brief Thread function to stop while loop on user command [Ctrl-D]
 */
void loopSpin(bool* spin) {
    std::string str;
    while (*spin) {
        *spin = bool(std::cin >> str);
    }
}

int main() {
    int sock;
    struct sockaddr_in server_address;

    // Create a UDP socket
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed!" << std::endl;
        return -1;
    }

    // Configure the server address
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(12345); // Port number for receiving side
    server_address.sin_addr.s_addr = inet_addr("192.168.2.2"); // Replace with your receiver's IP address

    // Initialize Research Interface
    harmony::ResearchInterface info;
    if (!info.init()) {
        std::cerr << "Failed to initialize research interface" << std::endl;
        return -1;
    }

    // Setup loop timing
    double fs = 200.0;
    double T_s = 1.0 / fs;           // time step in seconds
    double T_us = T_s * 1e6;         // time step in microseconds

    char logIndicator[4]{'-', '\\', '|', '/'};
    int i = 0;
    bool spin = true;

    std::thread spinThread(loopSpin, &spin);
    std::cout << "Enter [Ctrl-D] to stop recording.\n";

    while (spin) {
        // Get joint states
        auto rightJointStates = info.joints().rightArm.getOrderedStates();
        auto leftJointStates  = info.joints().leftArm.getOrderedStates();

        // Pack data: 7 joints × 2 values (angle, torque) × 2 arms = 28 doubles
        double jointData[28];
        for (int j = 0; j < harmony::armJointCount; ++j) {
            jointData[j * 2]     = rightJointStates[j].position_rad;
            jointData[j * 2 + 1] = rightJointStates[j].torque_Nm;

            jointData[(harmony::armJointCount + j) * 2]     = leftJointStates[j].position_rad;
            jointData[(harmony::armJointCount + j) * 2 + 1] = leftJointStates[j].torque_Nm;
        }

        // Send UDP packet
        int send_result = sendto(sock, jointData, sizeof(jointData), 0,
                                 (struct sockaddr*)&server_address, sizeof(server_address));
        if (send_result < 0) {
            std::cerr << "Failed to send data!" << std::endl;
            break;
        }

        std::cout << "Sending " << logIndicator[i++ % 4] << '\r';
        std::cout.flush();
        usleep(static_cast<useconds_t>(T_us));
    }

    spinThread.join();
    close(sock);
    return 0;
}
