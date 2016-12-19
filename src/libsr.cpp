#include "libsr.h"
#include "kernels.h"
#include "add.h"

using namespace libsr::kernels;

namespace libsr
{
    int add(int a, int b) {
        int result;

        run<kernels::add>(a, b, &result);
        return result;
    }
}