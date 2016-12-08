#include "gtest/gtest.h"
#include "libsr.h"
#include "libsr_config.h"

TEST(micro_flyweight, add)
{
    auto c = libsr::add(5, 4);
    EXPECT_EQ(c, 9);
}
