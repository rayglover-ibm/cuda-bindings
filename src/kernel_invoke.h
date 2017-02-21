#pragma once

#include "types.h"
#include "kernel.h"

#include <memory>
#include <type_traits>

namespace cufoo {
namespace kernel
{
    /* Kernel output traits ------------------------------------------------ */

    namespace detail
    {
        template <typename R> struct result_traits
        {
            using output_type = variant<error_code, R>;
            using public_type = maybe<R>;

            static error_code get_status(const output_type& s) {
                return s.is<error_code>() ? s.get<error_code>() : error_code::NONE;
            }
        };
        template <typename R> struct result_traits<variant<error_code, R>>
        {
            using output_type = variant<error_code, R>;
            using public_type = maybe<R>;

            static error_code get_status(const output_type& s) {
                return s.is<error_code>() ? s.get<error_code>() : error_code::NONE;
            }
        };
        template <> struct result_traits<void>
        {
            using output_type = error_code;
            using public_type = failable;

            static error_code get_status(const output_type& s) { return s; }
        };
        template <> struct result_traits<error_code>
        {
            using output_type = error_code;
            using public_type = failable;

            static error_code get_status(const output_type& s) { return s; }
        };
    }

    template <typename Op, typename... Args>
    using result_traits =
        typename detail::result_traits<
            decltype(Op::run<compute_mode::AUTO>(std::declval<Args>()...))
            >;

    template <typename Op, typename... Args>
    using result = typename result_traits<Op, Args...>::output_type;


    /* Runtime kernel selection -------------------------------------------- */

    template <compute_mode M, typename = void>
    struct control;

    /*  Used when the compute_mode is enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, std::enable_if_t< compute_traits<M>::enabled >>
    {
        template <typename Runner, typename Kernel, typename... Args>
        static auto call(Runner& r, Args... args)
            -> result<Kernel, Args...>
        {
            if (!r.begin(M)) { return error_code::CANCELLED; }
            auto s = r.apply<M, Args...>(std::forward<Args>(args)...);
            r.end(result_traits<Kernel, Args...>::get_status(s));
            return s;
        }
    };

    /*  Used when the compute_mode isn't enabled at compilation-time  */
    template <compute_mode M>
    struct control<M, std::enable_if_t< !compute_traits<M>::enabled >>
    {
        template <typename Runner, typename Kernel, typename... Args>
        static auto call(Runner& r, Args... args)
            -> result<Kernel, Args...>
        {
            return error_code::COMPUTE_MODE_DISABLED;
        }
    };

    /*  Specialization for AUTO: determines compute_mode at runtime  */
    template <>
    template <typename Runner, typename Kernel, typename... Args>
    auto control<compute_mode::AUTO>::call(Runner& r, Args... args)
        -> result<Kernel, Args...>
    {
        typename result<Kernel, Args...> s = error_code::KERNEL_NOT_DEFINED;

        /* Attempt to run cuda kernel */
        if (compute_traits<compute_mode::CUDA>::available())
        {
            s = control<compute_mode::CUDA>::call<Runner, Kernel, Args...>(
                    r, std::forward<Args>(args)...);
        }
        /* Attempt/fallback to run cpu kernel */
        if (s == error_code::KERNEL_NOT_DEFINED &&
            compute_traits<compute_mode::CPU>::available())
        {
            s = control<compute_mode::CPU>::call<Runner, Kernel, Args...>(
                    r, std::forward<Args>(args)...);
        }
        return s;
    }

    /* kernel runner ------------------------------------------------------- */

    template <typename K>
    struct runner
    {
        using traits = typename K::traits;

        bool begin(compute_mode m) { return true; }
        void end(error_code s) {}

        template <
            compute_mode M, typename... Args,
            std::enable_if_t< !K::template supports<M>::value, int> = 0
            >
        auto apply(Args... args) -> result<K, Args...> {
            return error_code::KERNEL_NOT_DEFINED;
        }

        template <
            compute_mode M, typename... Args,
            std::enable_if_t< K::template supports<M>::value, int> = 0
            >
        auto apply(Args... args) -> result<K, Args...> {
            return K::template run<M>(std::forward<Args>(args)...);
        }
    };

    template <typename K>
    struct log_runner : public runner<K>
    {
        log_runner(std::ostream* out) : m_out(out) {}

        bool begin(compute_mode m)
        {
            *m_out << '[' << traits::name << "] mode="
                   << to_str(m) << std::endl;

            return true;
        }

        void end(error_code s)
        {
            *m_out << '[' << traits::name << "] status="
                   << to_str(s) << std::endl;
        }

        private: std::ostream* m_out;
    };

    /* public api ---------------------------------------------------------- */

    namespace detail
    {
        template <typename R>
        maybe<R> convert(variant<error_code, R>&& r)
        {
            struct cvt {
                maybe<R> operator()(error_code s)  const { return to_str(s); }
                maybe<R> operator()(R& result) const { return std::move(result); }
            };
            return mapbox::util::apply_visitor(cvt{}, r);
        }

        failable convert(error_code r)
        {
            return r == error_code::NONE ?
                failable() : failable{ to_str(r) };
        };
    }

    template <
        typename K,
        compute_mode M = compute_mode::AUTO,
        typename Runner,
        typename... Args
        >
    typename result_traits<K, Args...>::public_type run_with(
        Runner& r, Args... args)
    {
        return detail::convert(
            control<M>::template call<Runner, K>(r, std::forward<Args>(args)...)
            );
    }

    template <
        typename K,
        compute_mode M = compute_mode::AUTO,
        typename... Args
        >
    typename result_traits<K, Args...>::public_type run(
        Args... args)
    {
        runner<K> r;

        return detail::convert(
            control<M>::template call<runner<K>, K>(r, std::forward<Args>(args)...)
            );
    }
}
}