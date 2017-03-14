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

#pragma once

#include "types.h"
#include "kernel.h"

#include <gsl.h>

namespace mwe {
namespace kernels
{
    using kernel::compute_mode;
    using kernel::error_code;
    
    /*  Declare a kernel, 'add' which is overloaded to operate
     *  on single or array-like inputs.
     *  
     *  We also declare the compute modes (CPU/GPU) which 
     *  this kernel will support.
     */
    KERNEL_DECL(add,
        compute_mode::CPU, compute_mode::CUDA)
    {
        template <compute_mode> static variant<int, error_code> op(
            int a, int b
            );
        
        template <compute_mode> static error_code op(
            const gsl::span<int> a, const gsl::span<int> b, gsl::span<int> result
            );
    };


}
}