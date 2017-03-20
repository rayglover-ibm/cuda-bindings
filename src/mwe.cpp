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

#include "mwe.h"
#include "kernels/add.h"

#include <kernelpp/kernel_invoke.h>

#include <gsl.h>
#include <type_traits>
#include <iostream>

using namespace kernelpp;

namespace mwe
{
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