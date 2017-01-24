#pragma once

#include "kernel.h"
#include <gsl.h>

namespace cufoo {
namespace kernels
{
    using kernel::compute_mode;
    using kernel::status;
    
    /*  Declare a kernel, 'add' which is overloaded to operate
     *  on single or array-like inputs.
     *  
     *  We also declare the compute modes (CPU/GPU) which 
     *  this kernel will support.
     */
    KERNEL_DECL(add,
        compute_mode::CPU, compute_mode::CUDA)
    {
        template<compute_mode> static status run(
            int a, int b, int* result
            );
        
        template<compute_mode> static status run(
            gsl::span<int> a, gsl::span<int> b, gsl::span<int> result
            );
    };
}
}