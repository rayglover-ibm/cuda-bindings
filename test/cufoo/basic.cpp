#include "gtest/gtest.h"
#include "cufoo.h"
#include "cufoo_config.h"

TEST(cufoo, add)
{
    auto c = cufoo::add(5, 4);
    EXPECT_EQ(c, 9);
}
