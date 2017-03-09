#pragma once_
#include "cufoo.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace util
{
    template<typename T>
    gsl::span<T> as_span(const pybind11::buffer_info& info) {
        return gsl::span<T>{ reinterpret_cast<T*>(info.ptr), info.size };
    }

    template<typename R>
    bool try_throw(const cufoo::maybe<R>& r)
    {
        if (r.is<cufoo::error>()) throw std::runtime_error(r.get<cufoo::error>().data());
        return false;
    }

    inline
    bool try_throw(const cufoo::status& r)
    {
        if (r) throw std::runtime_error(r.get().data());
        return false;
    }
}


/*  pybind11 type converters ----------------------------------------------- */


namespace pybind11 {
namespace detail
{
    /*
     *  Conversion to handle from a cufoo::maybe<T>; will
     *  throw an exception if the incoming cufoo::maybe<T> is
     *  in an error state.
     */
    template <typename T>
    class type_caster<cufoo::maybe<T>> {
    public:
        static handle cast(
            cufoo::maybe<T> &&src, return_value_policy policy, handle parent)
        {
            util::try_throw(src);
            return type_caster<T>::cast(src.get<T>(), policy, parent);
        }
        static PYBIND11_DESCR name() { return type_caster_base<T>::name(); }
    };
}
}