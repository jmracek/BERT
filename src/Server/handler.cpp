#include <netinet/in.h> 
#include <sys/socket.h>
#include <unistd.h> 

#include <memory>
#include <utility>
#include <iostream>

#include "handler.hpp"

// RequestHandler //
SUCCESS_CODE RequestHandler::getBytesFromSocket(int sock_fd, char *buffer, size_t buff_size, int n_bytes) {
    if (n_bytes > buff_size) return SUCCESS_CODE::BUFFER_SIZE_ERROR;

    int bytes_read = 0;
    int current_read = 0;
    while (bytes_read < n_bytes) {
        current_read = read(sock_fd, reinterpret_cast<void *>(&buffer[bytes_read]), n_bytes - bytes_read);
        if (current_read < 0) {
            std::cout << "[ERROR]: Could not read bytes from socket. Closing connection." << std::endl;
            return SUCCESS_CODE::READ_ERROR;
        }
        bytes_read += current_read;
    }

    return SUCCESS_CODE::SUCCESS;
}

// NewConnectionHandler //
NewConnectionHandler::NewConnectionHandler(
    int sock_fd, 
    std::unique_ptr<sockaddr_t>&& client_addr,
    std::shared_ptr<ObjectPool<Buffer>> buffer_src
): RequestHandler(sock_fd, std::move(client_addr)) {
    buffer_source = buffer_src;
}

void NewConnectionHandler::handle(void) {
    SUCCESS_CODE err;
    int message_len;
    Buffer* buffer = this->buffer_source->alloc();
    RequestType req;
    std::unique_ptr<RequestHandler> handler;
    
    // Get the message length
    err = getBytesFromSocket(NewConnectionHandler::sock_fd, reinterpret_cast<char *>(&message_len), sizeof(int), sizeof(int));
    if (err != SUCCESS) {
        std::cout << "[ERROR]: Could not read message length from socket. Closing connection." << std::endl;
        return;
    }
    // Read message type
    err = getBytesFromSocket(NewConnectionHandler::sock_fd, reinterpret_cast<char *>(&req), sizeof(RequestType), 1);
    if (err != SUCCESS) {
        std::cout << "[ERROR]: Could not read message type from socket. Closing connection." << std::endl;
        return;
    }

    // Read message bytes
    err = getBytesFromSocket(NewConnectionHandler::sock_fd, reinterpret_cast<char *>(buffer), sizeof(*buffer), message_len);
    if (err != SUCCESS) {
        std::cout << "[ERROR]: Could not read message bytes from socket. Closing connection." << std::endl;
        return;
    }
    
    // Decide which action to perform based on the message that was received
    switch (req) {
    case BATCH_INFERENCE:
        handler = std::make_unique<BatchInferenceRequestHandler>(
            this->sock_fd, 
            std::move(this->addr),
            message_len,
            buffer
        );
        break;
    case CACHE_UPDATE:
        break;
    case NEW_MODEL:
        break;
    case DELETE_MODEL:
        break;
    case SET_PROPERTIES:
        break;
    default:
        break;
    }  
    
    handler->handle();
    buffer_source->free(buffer); 
}

// BatchInferenceRequestHandler
BatchInferenceRequestHandler::BatchInferenceRequestHandler(
    int sock_fd, 
    std::unique_ptr<sockaddr_t> client_addr, 
    size_t message_length, 
    Buffer* buff
): RequestHandler(sock_fd, std::move(client_addr)) {
    this->buff = buff;
    this->message_length = message_length;
}

void BatchInferenceRequestHandler::handle(void) {
    
    std::cout << "Hi from BatchInferenceRequestHandler" << std::endl;
}

/*
 * Design decisions I need to make now:
 *  1) BatchInferenceRequestProto - what does this look like?
 *  2) What does the system design for model inference look like?
 *      - Models
 *      - Model inputs
 *  3) 
 */

