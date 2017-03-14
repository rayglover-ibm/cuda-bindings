/*  Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#include "device_util.h"

#include <mutex>

namespace mwe {
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
                return;
            }
            if (n == 0) {
                fprintf(stderr, "[E] 0 cuda devices detected\n");
                return;
            }
            success = true;
        });

        return success;
    }
}
}