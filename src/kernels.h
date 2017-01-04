#pragma once

#include "cufoo_config.h"
#include "device_util.h"

#include <memory>
#include <type_traits>

namespace cufoo {
namespace kernels
{
    enum class status {
        SUCCESS = 1,
        KERNEL_UNAVILABLE,
        KERNEL_NOT_DEFINED,
        KERNEL_FAILED,
        CANCELLED
    };

    enum class compute_mode {
        AUTO = 1,
        CUDA,
        CPU
    };

    const char* to_str(const status s);
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

    /* KERNEL -------------------------------------------------------------- */

    template <typename Op>
    struct kernel_traits
    {
        static constexpr const char* name = "undefined";
        using parameters = std::tuple<>;
    };

    template <typename Op, typename... Args>
    struct kernel_specialization_check
    {
        using expect = typename kernel_traits<Op>::parameters;
        using recieved = std::tuple<Args...>;

        static constexpr bool value = std::is_same<expect, recieved>::value;
        static_assert(value, "Calls to and implementations of a kernal must match the kernel signature.");
    };

    template <typename Op>
    struct kernel
    {
        template <
            compute_mode M,
            typename... Args,
            bool b = kernel_specialization_check<Op, Args...>::value
            >
        static status run(Args...) {
            return status::KERNEL_NOT_DEFINED;
        }
    };

    /* CONTROL ------------------------------------------------------------- */

    namespace detail
    {
        compute_mode runtime_mode();
    }

    template <compute_mode M, typename = void>
    struct control;

    /*  Used when the compute_mode is enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, typename std::enable_if< compute_traits<M>::enabled >::type>
    {
        template <typename Runner, typename Op, typename... Args>
        static status call(Runner& r, Args... args)
        {
            if (!r.template begin<Op>(M)) { return status::CANCELLED; }
            auto s = kernel<Op>::template run<M, Args...>(std::forward<Args>(args)...);
            r.template end<Op>(s);
            return s;
        }
    };

    /*  Used when the compute_mode isn't enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, typename std::enable_if< !compute_traits<M>::enabled >::type>
    {
        template <typename Runner, typename Op, typename... Args>
        static status call(Runner& r, Args... args)
        {
            if (!r.template begin<Op>(M)) { return status::CANCELLED; }
            status s = status::KERNEL_UNAVILABLE;
            r.template end<Op>(s);
            return s;
        }
    };

    /*  Specialization for AUTO: determines compute_mode at runtime  */
    template <> template <typename Runner, typename Op, typename... Args>
    status control<compute_mode::AUTO>::call(Runner& r, Args... args)
    {
        status s;

        switch (detail::runtime_mode()) {
        case compute_mode::CPU:
            s = control<compute_mode::CPU>::call<Runner, Op, Args...>(
                    r, std::forward<Args>(args)...);
            break;

        case compute_mode::CUDA:
            s = control<compute_mode::CUDA>::call<Runner, Op, Args...>(
                    r, std::forward<Args>(args)...);
            break;

        default:
            s = status::KERNEL_UNAVILABLE;
        }
        return s;
    }

    /* RUNNER -------------------------------------------------------------- */

    struct runner
    {
        template<typename Op>
        bool begin(compute_mode m) { return true; }

        template<typename Op>
        void end(status s) {}
    };

    struct log_runner
    {
        template<typename Op>
        bool begin(compute_mode m)
        {
            using kt = kernel_traits<Op>;
            printf("[%s] mode=%s\n", kt::name, to_str(m));
            fflush(stdout);
            return true;
        }

        template<typename Op>
        void end(status s)
        {
            using kt = kernel_traits<Op>;
            printf("[%s] status=%s\n", kt::name, to_str(s));
            fflush(stdout);
        }
    };

    template <
        typename Op,
        compute_mode M = compute_mode::AUTO,
        typename... Args
        >
    status run(Args... args)
    {
        runner r;

        return control<M>::template call<runner, Op>(
                r, std::forward<Args>(args)...);
    }

    template <
        typename Op,
        compute_mode M = compute_mode::AUTO,
        typename... Args
        >
    status run_logged(Args... args)
    {
        log_runner r;

        return control<M>::template call<log_runner, Op>(
                r, std::forward<Args>(args)...);
    }


#define KERNEL_DECL(Name, ...) \
    struct Name;                                    \
    template <> struct kernel_traits< Name > {      \
        static constexpr const char* name = #Name;  \
        using parameters = std::tuple<__VA_ARGS__>; \
    }

#define KERNEL_IMPL(Name, Mode) \
    template <> template <> status kernel<Name>::run<Mode>

}
}