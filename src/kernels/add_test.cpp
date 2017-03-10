#include "gtest/gtest.h"

#include "cufoo.h"
#include "cufoo_config.h"

#include <array>
#include <vector>
#include <thread>

using std::array;

TEST(cufoo, add)
{
    auto c = cufoo::add(5, 4);
    EXPECT_EQ(c, 9);
}

TEST(cufoo, add_span)
{
    array<int, 6> a{ 1, 2, 3, 4,  5,  6 };
    array<int, 6> b{ 7, 8, 9, 10, 11, 12 };
    array<int, 6> c;

    cufoo::add(a, b, c);

    array<int, 6> d{ 8, 10, 12, 14, 16, 18 };
    EXPECT_EQ(c, d);
}

TEST(cufoo, add_span_multithreaded)
{
    const uint32_t M = 1024000;

    std::vector<int> a(M, 7);
    std::vector<int> b(M, 21);

    auto fn = [&](int)
    {
        std::vector<int> c(M, 0);
        cufoo::add(a, b, c);
        for (int x : c) EXPECT_EQ(x, 7 + 21);
    };

    const uint32_t N = 4;
    std::thread t[N];

    for (int i = 0; i < N; ++i)
        t[i] = std::thread(fn, i);

    for (int i = 0; i < N; ++i)
        t[i].join();
}
