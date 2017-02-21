#include "types.h"
#include "kernels/add.h"

namespace cufoo {
namespace kernels
{
    template <> variant<status, int> add::run<compute_mode::CPU>(
        int a, int b)
    {
        return a + b;
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