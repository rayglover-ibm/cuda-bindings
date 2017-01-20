#include "add.h"

namespace cufoo {
namespace kernels
{
    template <> status add::run<compute_mode::CPU>(
        int a, int b, int* c)
    {
        *c = a + b;
        return status::SUCCESS;
    }
}
}