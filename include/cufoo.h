#pragma once
#include <gsl.h>

namespace cufoo
{
    int add(int a, int b);
    
    void add(gsl::span<int> a, gsl::span<int> b, gsl::span<int> result);
}