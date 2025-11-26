#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define PORT 12345  // The port the server will listen on

int main() {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    char buffer[1024];
    socklen_t client_addr_len = sizeof(client_addr);

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return -1;
    }

    // Zero out the server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    // Set up the server address structure
    server_addr.sin_family = AF_INET; // IPv4
    server_addr.sin_addr.s_addr = INADDR_ANY; // Listen on all available interfaces
    server_addr.sin_port = htons(PORT); // Port number

    // Bind the socket to the server address
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        close(sockfd);
        return -1;
    }

    std::cout << "UDP server listening on port " << PORT << std::endl;

    while (true) {
        // Receive message from the client
        int len = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0, 
                           (struct sockaddr *)&client_addr, &client_addr_len);
        if (len < 0) {
            std::cerr << "Failed to receive message" << std::endl;
            continue;
        }
        buffer[len] = '\0';  // Null-terminate the received data

        std::string received_message(buffer);
        std::cout << "Received: " << received_message << std::endl;

        // Echo the message back to the sender
        if (sendto(sockfd, received_message.c_str(), received_message.length(), 0, 
                   (const struct sockaddr *)&client_addr, client_addr_len) < 0) {
            std::cerr << "Failed to send message" << std::endl;
        } else {
            std::cout << "Echoed back: " << received_message << std::endl;
        }
    }

    // Close the socket
    close(sockfd);
    return 0;
}
