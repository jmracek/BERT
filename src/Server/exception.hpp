#include <exception>

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

struct SigInt: public std::exception {
    const char* what() const throw() {
        return "Signal Interrupt";
    }
};

#endif
