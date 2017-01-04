#include "device_util.h"

#include <mutex>

namespace cufoo {
namespace device_util
{
    bool init_cudart(void)
    {
        static bool success{ false };

        static std::once_flag flag;

        std::call_once(flag, [&]() {
            /* assume the runtime is statically linked. Here we just make sure
               cuda exists and the context is ready. */
            int n;
            cudaError_t err = cudaGetDeviceCount(&n);

            if (CUDA_SUCCESS != err) {
                fprintf(stderr, "[E] cuda failed to initialize. Is cuda installed? code: %04d\n", err);
                success = false;
                return;
            }
            if (n == 0) {
                fprintf(stderr, "[E] 0 cuda devices detected\n");
                success = false;
                return;
            }
            success = true;
        });

        return success;
    }
}
}