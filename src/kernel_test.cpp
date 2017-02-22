#include "gtest/gtest.h"

#include "kernel.h"
#include "kernel_invoke.h"

#include <array>
#include <vector>

using namespace cufoo::kernel;

namespace
{
    KERNEL_DECL(foo, compute_mode::CPU)
    {
        template<compute_mode> static void run();
        template<compute_mode> static int  run(std::vector<float>&);
    };

    int void_calls = 0;

    template <> void foo::run<compute_mode::CPU>() {
        void_calls++;
    }

    template <> int foo::run<compute_mode::CPU>(std::vector<float>& v) {
        return v.size();
    }
}

TEST(kernel, call_void)
{
    ::void_calls = 0;
    cufoo::status err = run<foo>();

    EXPECT_FALSE(err);
    EXPECT_EQ(::void_calls, 1);

    err = run<foo, compute_mode::CPU>();

    EXPECT_FALSE(err);
    EXPECT_EQ(::void_calls, 2);
}

TEST(kernel, call_undefined)
{
    ::void_calls = 0;
    cufoo::status err = run<foo, compute_mode::CUDA>();

    EXPECT_TRUE(err);
    EXPECT_EQ(::void_calls, 0);
}

TEST(kernel, call_vector)
{
    ::void_calls = 0;

    std::vector<float> vec(5, 0);
    cufoo::maybe<int> result = run<foo>(vec);

    EXPECT_FALSE(result.is<cufoo::error>());
    EXPECT_EQ(result.get<int>(), 5);
}


namespace
{
    KERNEL_DECL(foo_2, compute_mode::CPU, compute_mode::CUDA) {
        template<compute_mode> static void run();
    };

    int cuda_calls = 0;
    int cpu_calls = 0;

    template <> void foo_2::run<compute_mode::CPU>()  { cpu_calls++; }
    template <> void foo_2::run<compute_mode::CUDA>() { cuda_calls++; }
}

TEST(kernel, call_cuda)
{
    using traits = compute_traits<compute_mode::CUDA>;

    EXPECT_FALSE(run<foo_2>());

    if (traits::enabled && traits::available()) {
        EXPECT_EQ(0, cpu_calls);
        EXPECT_EQ(1, cuda_calls);
    }
    else {
        EXPECT_EQ(1, cpu_calls);
        EXPECT_EQ(0, cuda_calls);
    }
}