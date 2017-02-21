#pragma once

#include <mapbox/variant.hpp>
#include <mapbox/optional.hpp>

namespace cufoo
{
    /* Types  -------------------------------------------------------------- */

    /* standard cufoo error type */
    using error = std::string;

    /* introduce a variant type */
    using mapbox::util::variant;

    /* an optional value of type T or an error */
    template <typename T> using maybe = variant<T, error>;

    /* an optional error */
    using status = mapbox::util::optional<error>;
}