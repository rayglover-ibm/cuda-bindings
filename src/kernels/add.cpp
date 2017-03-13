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

#include "types.h"
#include "kernels/add.h"

namespace cufoo {
namespace kernels
{
    template <> variant<int, error_code> add::op<compute_mode::CPU>(
        int a, int b)
    {
        return a + b;
    }

    template <> error_code add::op<compute_mode::CPU>(
        const gsl::span<int> a, const gsl::span<int> b, gsl::span<int> result)
    {
        const size_t N = a.length();
        
        if (N != b.length() || N != result.length())
            /* inputs have unequal shape */
            return error_code::KERNEL_FAILED;

        for (size_t i = 0u; i < N; i++) {
            result[i] = a[i] + b[i];
        }

        return error_code::NONE;
    }
}
}