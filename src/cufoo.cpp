#include "cufoo.h"
#include "kernel_invoke.h"
#include "kernels/add.h"

#include <gsl.h>
#include <type_traits>
#include <iostream>

namespace cufoo
{
    using namespace kernel;

    maybe<int> add(int a, int b) {
        return run<kernels::add>(a, b);
    }

    status add(gsl::span<int> a, gsl::span<int> b, gsl::span<int> result)
    {
        log_runner<kernels::add> log(&std::cout);

        /* force CPU for small inputs */
        return (a.size() < 1000) ?
              run_with<kernels::add, compute_mode::CPU>(log, a, b, result)
            : run_with<kernels::add>(log, a, b, result);
    }
}