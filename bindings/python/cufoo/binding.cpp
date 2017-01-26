/*
 *   For more pybind11 examples, see: 
 *   https://github.com/pybind/pybind11/tree/master/tests 
 */
#include "cufoo.h"
#include "cufoo_config.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace util
{
    template<typename T>
    gsl::span<T> as_span(const py::buffer_info& info) {
        return gsl::span<T>{ reinterpret_cast<T*>(info.ptr), info.size };
    }
}

namespace
{
    std::valarray<int> get_version() {
        return std::valarray<int>({ 
            cufoo_VERSION_MAJOR, cufoo_VERSION_MINOR, cufoo_VERSION_PATCH 
        });
    }

    py::array_t<int> add_all(py::array_t<int> input1, py::array_t<int> input2)
    {
        auto buf1 = input1.request(), buf2 = input2.request();

        if (buf1.ndim != 1 || buf2.ndim != 1)
            throw std::runtime_error("Number of dimensions must be one");

        auto result = py::array_t<int>(buf1.size);
        auto buf3 = result.request();

        cufoo::add(util::as_span<int>(buf1),
                   util::as_span<int>(buf2),
                   util::as_span<int>(buf3));
        
        return result;
    }
}

PYBIND11_PLUGIN(binding)
{
    py::module m("binding", "python binding example");

    m.def("version", &::get_version,
        "Module version");
    
    m.def("add", static_cast<int (*)(int, int)>(&cufoo::add),
        "A function which adds two numbers");
    
    m.def("add_all", &add_all, 
        "A function which adds two 1d arrays of equal size together");

    return m.ptr();
}