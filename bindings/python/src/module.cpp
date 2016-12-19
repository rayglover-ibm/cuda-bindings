#include <pybind11/pybind11.h>
#include "libsr.h"

namespace py = pybind11;

PYBIND11_PLUGIN(libsrpy) {
    py::module m("libsrpy", "python binding example");

    m.def("add", static_cast<int (*)(int, int)>(&libsr::add), "A function which adds two numbers");

    return m.ptr();
}