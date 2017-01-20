#pragma once

#include "cufoo_config.h"
#include "device_util.h"

#include <memory>
#include <type_traits>

namespace cufoo {
namespace kernel
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

    namespace detail
    {
        template<compute_mode T, compute_mode U>
        constexpr bool eq() { return T == U; }

        template<compute_mode... Ts>
        struct has_mode : std::false_type {};

        template<compute_mode T0, compute_mode T1, compute_mode... Ts>
        struct has_mode<T0, T1, Ts...> :
            std::integral_constant<bool, eq<T0, T1>() || has_mode<T0, Ts...>::value>
        {};
    }
    
    /* KERNEL -------------------------------------------------------------- */

    template<typename Op, typename Traits, compute_mode... Modes>
    struct impl
    {
        using traits = Traits;

        template<
            compute_mode M, typename... Args,
            std::enable_if_t< detail::has_mode<M, Modes... >::value, int> = 0
            >
        static auto apply(Args... args) {
            return Op::template run<M>(std::forward<Args>(args)...);
        }

        template<
            compute_mode M, typename... Args,
            std::enable_if_t< !detail::has_mode<M, Modes... >::value, int> = 0
            >
        static auto apply(Args... args) {
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
    struct control<M, std::enable_if_t< compute_traits<M>::enabled >>
    {
        template <typename Runner, typename Kernel, typename... Args>
        static status call(Runner& r, Args... args)
        {
            if (!r.template begin<Kernel>(M)) { return status::CANCELLED; }
            status s = Kernel::template apply<M, Args...>(std::forward<Args>(args)...);
            r.template end<Kernel>(s);
            return s;
        }
    };

    /*  Used when the compute_mode isn't enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, std::enable_if_t< !compute_traits<M>::enabled >>
    {
        template <typename Runner, typename Kernel, typename... Args>
        static status call(Runner& r, Args... args)
        {
            if (!r.template begin<Kernel>(M)) { return status::CANCELLED; }
            status s = status::KERNEL_UNAVILABLE;
            r.template end<Kernel>(s);
            return s;
        }
    };

    /*  Specialization for AUTO: determines compute_mode at runtime  */
    template <> template <typename Runner, typename Kernel, typename... Args>
    status control<compute_mode::AUTO>::call(Runner& r, Args... args)
    {
        status s;

        switch (detail::runtime_mode()) {
        case compute_mode::CPU:
            s = control<compute_mode::CPU>::call<Runner, Kernel, Args...>(
                    r, std::forward<Args>(args)...);
            break;

        case compute_mode::CUDA:
            s = control<compute_mode::CUDA>::call<Runner, Kernel, Args...>(
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
        template<typename Kernel>
        bool begin(compute_mode m)
        {
            using kt = typename Kernel::traits;
            printf("[%s] mode=%s\n", kt::name, to_str(m));
            fflush(stdout);
            return true;
        }

        template<typename Kernel>
        void end(status s)
        {
            using kt = typename Kernel::traits;
            printf("[%s] status=%s\n", kt::name, to_str(s));
            fflush(stdout);
        }
    };

    template <
        typename Kernel,
        compute_mode M = compute_mode::AUTO,
        typename... Args
        >
    status run(Args... args)
    {
        runner r;

        return control<M>::template call<runner, Kernel>(
                r, std::forward<Args>(args)...);
    }

    template <
        typename Kernel,
        compute_mode M = compute_mode::AUTO,
        typename... Args
        >
    status run_logged(Args... args)
    {
        log_runner r;

        return control<M>::template call<log_runner, Kernel>(
                r, std::forward<Args>(args)...);
    }

    #define KERNEL_DECL(Name, ...) \
        struct Name_traits_ {                            \
            using return_type = ::cufoo::kernel::status; \
            static constexpr const char* name = #Name;   \
        };                                               \
        struct Name : ::cufoo::kernel::impl<Name, Name_traits_, __VA_ARGS__ >
}
}

