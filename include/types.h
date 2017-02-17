#pragma once

#include <mapbox/variant.hpp>
#include <mapbox/optional.hpp>

namespace cufoo
{
    /* Types  -------------------------------------------------------------- */

    /* standard cufoo error type */
    using error = std::string;

    /* an optional value of type T or an error */
    template <typename T> using maybe = mapbox::util::variant<T, error>;

    /* an optional error */
    using failable = mapbox::util::optional<error>;
}