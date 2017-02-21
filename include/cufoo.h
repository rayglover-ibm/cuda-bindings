#pragma once

#include "types.h"
#include <gsl.h>

namespace cufoo
{
    /* Operations  --------------------------------------------------------- */
    
    maybe<int> add(int a, int b);
    
    status add(const gsl::span<int> a,
               const gsl::span<int> b,
                     gsl::span<int> result
              );
}