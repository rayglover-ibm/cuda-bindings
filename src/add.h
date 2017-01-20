#pragma once
#include "kernel.h"

namespace cufoo {
namespace kernels
{
    using namespace kernel;
    
    KERNEL_DECL(add,
        compute_mode::CPU, compute_mode::CUDA)
    {
        template<compute_mode>
        static status run(int, int, int*);
    };
    
    // struct add_traits {
    //     static constexpr const char* name = "blah";
    // };
    // 
    // struct add : kernel::impl<add, add_traits,
    //     kernel::compute_mode::CPU, kernel::compute_mode::CUDA
    //     >
    // {
    //     template<kernel::compute_mode>
    //     static kernel::status run(int, int, int*);
    // };
}
}