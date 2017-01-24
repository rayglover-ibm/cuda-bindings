#include "gtest/gtest.h"
#include "cufoo.h"
#include "cufoo_config.h"

#include <array>

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
