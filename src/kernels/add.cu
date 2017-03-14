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

#include "add.h"
#include "kernel.h"
#include "types.h"
#include "device_util.h"

#include <gsl.h>
#include <stdio.h>

namespace
{
    __global__ void add(int a, int b, int* result) {
        *result = a + b;
    }

    __global__ void add_span(
        gsl::span<int> a, gsl::span<int> b, gsl::span<int> result
        )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < result.size()) {
            result[idx] = a[idx] + b[idx];
        }
    }
}

namespace mwe {
namespace kernels
{
    using device_util::device_ptr;

    template <> variant<int, error_code> add::op<compute_mode::CUDA>(
        int a, int b
        )
    {
        device_ptr<int> dev_c;

        ::add<<< 1, 1 >>>(a, b, dev_c.get());
        if (!checkCudaLastError()) return error_code::KERNEL_FAILED;

        int c;
        dev_c.copy_to({ &c, 1 });
        
        return c;
    }

    template <> error_code add::op<compute_mode::CUDA>(
        const gsl::span<int> a, const gsl::span<int> b, gsl::span<int> result
        )
    {
        size_t N = result.size();

        if (N != a.length() || N != b.length())
            return error_code::KERNEL_FAILED;

        device_ptr<int> dev_a(a);
        device_ptr<int> dev_b(b);
        device_ptr<int> dev_result(N);

        int blockSize, minGridSize, gridSize;

        if (!checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, ::add_span, 0, 0)))
            return error_code::KERNEL_FAILED;

        /* Round up according to array size */
        gridSize = ((int) N + blockSize - 1) / blockSize;

        ::add_span<<< gridSize, blockSize >>>(
            dev_a.span(), dev_b.span(), dev_result.span());

        if (!checkCudaLastError()) return error_code::KERNEL_FAILED;

        dev_result.copy_to(result);

        return error_code::NONE;
    }
}
}