#pragma once

#include "types.h"
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
        template <compute_mode> static variant<status, int> run(
            int a, int b
            );
        
        template <compute_mode> static status run(
            const gsl::span<int> a, const gsl::span<int> b, gsl::span<int> result
            );
    };


}
}