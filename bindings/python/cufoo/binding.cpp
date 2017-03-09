/*
 *   For more pybind11 examples, see:
 *   https://github.com/pybind/pybind11/tree/master/tests
 */
#include "cufoo.h"
#include "cufoo_config.h"
#include "binding_util.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace
{
    std::valarray<int> get_version() {
        return std::valarray<int>({
            cufoo_VERSION_MAJOR, cufoo_VERSION_MINOR, cufoo_VERSION_PATCH
        });
    }

    py::array_t<int> add_all(py::array_t<int> a, py::array_t<int> b)
    {
        auto buf1 = a.request(), buf2 = b.request();

        if (buf1.ndim != 1 || buf2.ndim != 1)
            throw std::runtime_error("Number of dimensions must be one");

        auto result = py::array_t<int>(buf1.size);

        util::try_throw(cufoo::add(
            util::as_span<int>(buf1),
            util::as_span<int>(buf2),
            util::as_span<int>(result.request())));

        return result;
    }
}

PYBIND11_PLUGIN(binding)
{
    py::module m("binding", "python binding example");

    m.def("version", &::get_version,
        "Module version");

    m.def("add", [](int a, int b) { return cufoo::add(a, b); },
        "A function which adds two numbers");

    m.def("add", &::add_all,
        "A function which adds two 1d arrays of equal size together");

    return m.ptr();
}