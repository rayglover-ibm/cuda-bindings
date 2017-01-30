#include "kernels/add.h"

namespace cufoo {
namespace kernels
{
    template <> status add::run<compute_mode::CPU>(
        int a, int b, int* c)
    {
        *c = a + b;
        return status::SUCCESS;
    }

    template <> status add::run<compute_mode::CPU>(
        const gsl::span<int> a, const gsl::span<int> b, gsl::span<int> result)
    {
        const size_t N = a.length();
        
        if (N != b.length() || N != result.length())
            /* inputs have unequal shape */
            return status::KERNEL_FAILED;

        for (size_t i = 0u; i < N; i++) {
            result[i] = a[i] + b[i];
        }

        return status::SUCCESS;
    }
}
}