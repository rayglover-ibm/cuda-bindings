#pragma once
#include <gsl.h>

namespace cufoo
{
    int add(int a, int b);
    
    void add(const gsl::span<int> a,
             const gsl::span<int> b,
                   gsl::span<int> result
            );
}