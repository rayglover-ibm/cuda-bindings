#pragma once

#include "cufoo_config.h"
#include "device_util.h"

#include <memory>
#include <type_traits>

namespace cufoo {
namespace kernel
{
    /* the suppported compute modes */
    enum class compute_mode { AUTO = 1, CUDA, CPU };

    enum class error_code : uint8_t
    {
        /* The kernel ran and completed successfully */
        NONE = 0,

        /* The kernel was invoked with a compute mode that wasn't
           enabled at compile time or is unavailable at run-time  */
        COMPUTE_MODE_DISABLED,

        /* The kernel does not support being invoked with the
           specified compute mode or arguments */
        KERNEL_NOT_DEFINED,

        /* The kernel was invoked but failed during execution. */
        KERNEL_FAILED,

        /* The kernel was invoked but was cancelled before it
           began executing. */
        CANCELLED
    };

    const char* to_str(const error_code s);
    const char* to_str(const compute_mode m);

    template <compute_mode>
    struct compute_traits {
        static constexpr bool enabled = true;
        static bool available() { return true; }
    };

#if !defined(cufoo_WITH_CUDA)
    template <>
    struct compute_traits<compute_mode::CUDA> {
        static constexpr bool enabled = false;
        static bool available() { return false; }
    };
#else
    template <>
    struct compute_traits<compute_mode::CUDA> {
        static constexpr bool enabled = true;
        static bool available() { return device_util::init_cudart(); }
    };
#endif

    namespace detail
    {
        template <compute_mode T, compute_mode U>
        constexpr bool eq() { return T == U; }

        template <compute_mode... Ts>
        struct has_mode : std::false_type {};

        template <compute_mode T0, compute_mode T1, compute_mode... Ts>
        struct has_mode<T0, T1, Ts...> :
            std::integral_constant<bool, eq<T0, T1>() || has_mode<T0, Ts...>::value>
        {};
    }

    /* Kernel declarations ------------------------------------------------- */

    template<typename Traits, compute_mode... Modes>
    struct impl
    {
        template <compute_mode M>
        using supports = detail::has_mode<M, Modes... >;

        using traits = Traits;
    };

    #define KERNEL_DECL(Name, ...) \
        struct Name_traits_ {                            \
            static constexpr const char* name = #Name;   \
        };                                               \
        struct Name : ::cufoo::kernel::impl<Name_traits_, __VA_ARGS__ >

}
}

