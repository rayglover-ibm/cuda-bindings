/*
 *   For more pybind11 examples, see: 
 *   https://github.com/pybind/pybind11/tree/master/tests 
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libsr.h"
#include "libsr_config.h"

namespace py = pybind11;

namespace {
    std::valarray<int> get_version() {
        return std::valarray<int>({ 
            libsr_VERSION_MAJOR, libsr_VERSION_MINOR, libsr_VERSION_PATCH 
        });
    }
}

PYBIND11_PLUGIN(libsrpy) {
    py::module m("libsrpy", "python binding example");

    m.def("version", &::get_version, "module version");
    m.def("add", static_cast<int (*)(int, int)>(&libsr::add), "A function which adds two numbers");

    return m.ptr();
}