#include "cufoo.h"
#include "kernels.h"
#include "add.h"

using namespace cufoo::kernels;

namespace cufoo
{
    int add(int a, int b) {
        int result;

        run_logged<kernels::add>(a, b, &result);
        return result;
    }
}