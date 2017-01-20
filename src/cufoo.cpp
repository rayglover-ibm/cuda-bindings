#include "cufoo.h"
#include "kernel.h"
#include "add.h"

namespace cufoo
{
    int add(int a, int b) {
        int result;

        kernel::run_logged<kernels::add>(a, b, &result);
        return result;
    }
}