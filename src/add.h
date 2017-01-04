#pragma once
#include "kernels.h"

namespace cufoo {
namespace kernels
{
    KERNEL_DECL(add, int, int, int*);

    KERNEL_IMPL(add, compute_mode::CPU)(int, int, int*);

    KERNEL_IMPL(add, compute_mode::CUDA)(int, int, int*);
}
}