
#ifndef HANDLERS_HPP
#define HANDLERS_HPP

#include <memory>
#include "memory.hpp"
#include "buffer.hpp"

using sockaddr_t = struct sockaddr_in;

enum SUCCESS_CODE {
    SUCCESS,
    BUFFER_SIZE_ERROR,
    READ_ERROR
};

enum RequestType {
    BATCH_INFERENCE,
    CACHE_UPDATE,
    NEW_MODEL,
    DELETE_MODEL,
    SET_PROPERTIES
};

class RequestHandler {
protected:
    std::unique_ptr<sockaddr_t> addr;
    const int sock_fd;
    SUCCESS_CODE getBytesFromSocket(int sock_fd, char* buffer, size_t buff_size, int n_bytes);

public:
    RequestHandler(int client_sock_fd, std::unique_ptr<sockaddr_t>&& client_addr):
        sock_fd(client_sock_fd),
        addr(std::move(client_addr)) {
    }
    virtual void handle(void) = 0;
};

class NewConnectionHandler: public RequestHandler {
private:
    std::shared_ptr<ObjectPool<Buffer>> buffer_source;
public:
    NewConnectionHandler(int sock_fd, std::unique_ptr<sockaddr_t>&& client_addr, std::shared_ptr<ObjectPool<Buffer>> buffer_src); 
    void handle(void) override;
};

class BatchInferenceRequestHandler: public RequestHandler {
private:
    size_t message_length;
    Buffer* buff;
public:
    BatchInferenceRequestHandler(int sock_fd, std::unique_ptr<sockaddr_t> client_addr, size_t message_length, Buffer* buff);
    void handle(void) override;
};

#endif
