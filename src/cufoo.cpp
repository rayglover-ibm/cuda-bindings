#include "cufoo.h"
#include "kernel.h"
#include "add.h"

#include "gsl.h"

namespace cufoo
{
    int add(int a, int b) {
        int result;

        kernel::run_logged<kernels::add>(a, b, &result);
        return result;
    }

    void add(gsl::span<int> a, gsl::span<int> b, gsl::span<int> result) {
        kernel::run_logged<kernels::add>(a, b, result);
    }
}