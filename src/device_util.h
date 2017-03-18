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

#pragma once

#include <gsl.h>
#include <memory>

#if defined(mwe_WITH_CUDA)
#   define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#   define checkCudaLastError()  __checkLastCudaError (__FILE__, __LINE__)
#endif

namespace mwe {
namespace device_util
{
    /*  Initialize the cuda runtime, and ensure at least
     *  one device is available.
     */
    bool init_cudart();

    /*  `device_ptr<T>` is a smart pointer which owns and manages one
     *  or more objects of type `T` in contiguous memory on a CUDA device,
     *  and disposes of these objects when it goes out of scope.
     */
    template<typename T> class device_ptr final
    {
        struct cuda_deleter;
        std::unique_ptr<T, cuda_deleter> m_ptr;
        size_t m_size;

      public:
        device_ptr();
        device_ptr(size_t n);
        device_ptr(device_ptr<T>&& other);
        device_ptr(gsl::span<T> span);

        gsl::span<T> span();
        const gsl::span<T> span() const;

        T* get();

        void copy_from(const gsl::span<T> from);
        void copy_to(gsl::span<T> to) const;
    };
}
}

#if defined(mwe_WITH_CUDA) && defined(__CUDACC__)
#   include "device_util-inl.cuh"
#endif
