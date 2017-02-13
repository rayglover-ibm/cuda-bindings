#include "add.h"
#include "kernel.h"
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

namespace cufoo {
namespace kernels
{
    using device_util::device_ptr;

    template <> status add::run<compute_mode::CUDA>(
        int a, int b, int* c
        )
    {
        device_ptr<int> dev_c;

        ::add<<< 1, 1 >>>(a, b, dev_c.data());
        if (!checkCudaLastError()) return status::KERNEL_FAILED;

        dev_c.copy_to({ c, 1 });

        return status::SUCCESS;
    }

    template <> status add::run<compute_mode::CUDA>(
        const gsl::span<int> a, const gsl::span<int> b, gsl::span<int> result
        )
    {
        size_t N = result.size();

        if (N != a.length() || N != b.length())
            return status::KERNEL_FAILED;

        device_ptr<int> dev_a(a);
        device_ptr<int> dev_b(b);
        device_ptr<int> dev_result(N);

        int blockSize, minGridSize, gridSize;

        if (!checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, ::add_span, 0, 0)))
            return status::KERNEL_FAILED;

        /* Round up according to array size */
        gridSize = ((int) N + blockSize - 1) / blockSize;

        ::add_span<<< gridSize, blockSize >>>(
            dev_a.span(), dev_b.span(), dev_result.span());

        if (!checkCudaLastError()) return status::KERNEL_FAILED;

        dev_result.copy_to(result);

        return status::SUCCESS;
    }
}
}