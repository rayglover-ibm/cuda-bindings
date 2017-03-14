/*  Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#include "gtest/gtest.h"

#include "mwe.h"
#include "mwe_config.h"

#include <array>
#include <vector>
#include <thread>

using std::array;

TEST(mwe, add)
{
    auto c = mwe::add(5, 4);
    EXPECT_EQ(c, 9);
}

TEST(mwe, add_span)
{
    array<int, 6> a{ 1, 2, 3, 4,  5,  6 };
    array<int, 6> b{ 7, 8, 9, 10, 11, 12 };
    array<int, 6> c;

    mwe::add(a, b, c);

    array<int, 6> d{ 8, 10, 12, 14, 16, 18 };
    EXPECT_EQ(c, d);
}

TEST(mwe, add_span_multithreaded)
{
    const uint32_t M = 1024000;

    std::vector<int> a(M, 7);
    std::vector<int> b(M, 21);

    auto fn = [&](int)
    {
        std::vector<int> c(M, 0);
        mwe::add(a, b, c);
        for (int x : c) EXPECT_EQ(x, 7 + 21);
    };

    const uint32_t N = 4;
    std::thread t[N];

    for (int i = 0; i < N; ++i)
        t[i] = std::thread(fn, i);

    for (int i = 0; i < N; ++i)
        t[i].join();
}
