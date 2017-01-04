#include "add.h"

namespace cufoo {
namespace kernels
{
    KERNEL_IMPL(add, compute_mode::CPU)(
        int a, int b, int* c)
    {
        *c = a + b;
        return status::SUCCESS;
    }
}
}