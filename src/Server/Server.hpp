#include <netinet/in.h> 
#include <sys/socket.h>
#include <unistd.h> 
#include <arpa/inet.h>

#include <iostream>
#include <memory>

#include "../Utils/ThreadPool.hpp"
#include "exception.hpp"
#include "Handler.hpp"

using sockaddr_t = struct sockaddr_in;

class Server {
private:
    static constexpr int ADDR_SIZE = sizeof(sockaddr_t);
    static constexpr int MAX_INCOMING_CONNECTIONS = 128;
    static constexpr int NUM_WORKER_THREADS = 32;
    int server_fd; 
    sockaddr_t address;
    std::shared_ptr<ThreadPool> pool;

public:
    Server(int port) {
        pool = std::make_shared<ThreadPool>(NUM_WORKER_THREADS);

        address.sin_family = AF_INET; 
        address.sin_addr.s_addr = INADDR_ANY; 
        address.sin_port = htons(port); 
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == 0) {
            std::cout << "[ERROR]: Creating server socket failed. Exiting." << std::endl;
            exit(1);
        }
        int bind_err = bind(server_fd, (const struct sockaddr *)&address, ADDR_SIZE);
        if (bind_err < 0) {
            std::cout << "[ERROR]: Could not bind to server socket. Exiting." << std::endl;
            exit(1);
        }
        int err = listen(server_fd, MAX_INCOMING_CONNECTIONS);
        if (err < 0) {
            std::cout << "[ERROR]: Unable to set server socket to listen. Exiting." << std::endl;
            exit(1);
        }
    }
    
    ~Server(void) { close(server_fd); }

    void run(void) {
        int sock_fd;
        socklen_t addr_len;

        pool->start();

        try {
            while(1) {
                auto client_ptr = std::make_unique<sockaddr_t>();
                sock_fd = accept(server_fd, (struct sockaddr *)client_ptr.get(), &addr_len);
                std::cout << "[INFO]: Received new connection from " 
                          << inet_ntoa(client_ptr->sin_addr)
                          << std::endl;
                pool->submit(
                    [](RequestHandler* handler) {
                        handler->handle();
                        delete handler;
                    },
                    new NewConnectionHandler(sock_fd, std::move(client_ptr))
                );
            }
        }
        catch (SigInt e) {
            std::cout << "[INFO]: Received termination signal.  Stopping server." << std::endl;
            pool->stop();
        }
    }
};


/*
 * What should we do when we receive a new connection?
 * 4 bytes = message length, followed by message_length bytes protobuf object, little endian
 * 1 byte = request type
 * N bytes = serialized pb object
 * Request = {
 *  api_key, 
 *  model_id,
 *  inputs
 * }
 *
 * Read the first 4 bytes and parse an int.
 * Read the number of bytes described by the int we just read
 * Serialize protobuf object from 

 * What sorts of requests do I want to allow?
 *  1) Model batch inference
 *  2) Create a new model
 *  3) Delete a model
 *  4) Modify properties of serving (caching behaviour, etc.)
 */

