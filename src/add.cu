#include "add.h"
#include "kernel.h"
#include "device_util.h"

#include <stdio.h>

namespace {
    __global__ void add(int a, int b, int *c) {
        *c = a + b;
    }
}

namespace cufoo {
namespace kernels
{
    template <> status add::run<compute_mode::CUDA>(
        int a, int b, int* c
        )
    {
        int *dev_c;
        checkCudaErrors(cudaMalloc((void**)&dev_c, sizeof(int)));

        ::add<<<1,1>>>(a, b, dev_c);
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("[E] %s", cudaGetErrorString(cudaGetLastError()));
            return status::KERNEL_FAILED;
        }

        cudaDeviceSynchronize();
        checkCudaErrors(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(dev_c));

        return status::SUCCESS;
    }
}
}