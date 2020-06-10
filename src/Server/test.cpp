#include <csignal>
#include <functional>

#include "exception.hpp"
#include "Server.hpp"

void sigint_handler(int signal) {
    throw SigInt();
}

int main(void) {
    std::signal(SIGINT, sigint_handler);

    Server serv(8080);
    serv.run();
    return 0;
}
