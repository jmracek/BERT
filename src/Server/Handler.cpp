#include <netinet/in.h> 
#include <sys/socket.h>
#include <unistd.h> 

#include <memory>
#include <utility>
#include <iostream>

#include "Handler.hpp"

#include "State/BatchInferenceWorkflowDispatcher.hpp"
#include "State/BatchInferenceWorkflow.hpp"

// RequestHandler //
SUCCESS_CODE RequestHandler::getBytesFromSocket(int sock_fd, char *buffer, size_t buff_size, size_t n_bytes) {
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
    std::unique_ptr<sockaddr_t>&& client_addr
): RequestHandler(sock_fd, std::move(client_addr)) {}

void NewConnectionHandler::handle(void) {
    SUCCESS_CODE err;
    RequestType req;
    std::unique_ptr<RequestHandler> handler;
    
    // Read message type
    err = getBytesFromSocket(this->sock_fd, reinterpret_cast<char *>(&req), sizeof(RequestType), 1);
    if (err != SUCCESS) {
        std::cout << "[ERROR]: Could not read message type from socket. Closing connection." << std::endl;
        return;
    }
    
    // Decide which action to perform based on the message that was received
    switch (req) {
    case BATCH_INFERENCE:
        handler = std::make_unique<BatchInferenceRequestHandler>(
            this->sock_fd, 
            std::move(this->addr)
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
        std::cout << "[WARNING]: Unrecognized message type. Ignoring current request" << std::endl;
        goto close;
        break;
    }  
    
    handler->handle();
    
    close:
        int close_error = close(this->sock_fd);
        if (close_error < 0) {
            std::cout << "[ERROR]: Could not close socket. Terminating." << std::endl;
            exit(1);
        }
}

// BatchInferenceRequestHandler
BatchInferenceRequestHandler::BatchInferenceRequestHandler(
    int sock_fd, 
    std::unique_ptr<sockaddr_t> client_addr
): RequestHandler(sock_fd, std::move(client_addr)) {}

void BatchInferenceRequestHandler::handle(void) {

    BatchInferenceWorkflow wf(this->sock_fd);
    BatchInferenceWorkflowDispatcher dispatcher;
    
    do {
        wf.accept(dispatcher);
    } while (!dispatcher.finished());

}
