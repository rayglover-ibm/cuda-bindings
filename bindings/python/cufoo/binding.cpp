/*
 *   For more pybind11 examples, see: 
 *   https://github.com/pybind/pybind11/tree/master/tests 
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cufoo.h"
#include "cufoo_config.h"

namespace py = pybind11;

namespace {
    std::valarray<int> get_version() {
        return std::valarray<int>({ 
            cufoo_VERSION_MAJOR, cufoo_VERSION_MINOR, cufoo_VERSION_PATCH 
        });
    }
}

PYBIND11_PLUGIN(binding) {
    py::module m("binding", "python binding example");

    m.def("version", &::get_version, "module version");
    m.def("add", static_cast<int (*)(int, int)>(&cufoo::add), "A function which adds two numbers");

    return m.ptr();
}