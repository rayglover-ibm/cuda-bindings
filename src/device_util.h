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

#if defined(__CUDACC__)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#define checkCudaLastError()  __checkLastCudaError (__FILE__, __LINE__)

inline bool __checkCudaErrors(CUresult err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        fprintf(stderr,
            "[E] CUDA Driver API error = %04d from file <%s:%i>.\n",
            err, file, line
            );
        fflush(stderr);
        return false;
    }
    return true;
}

inline bool __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        fprintf(stderr,
            "[E] CUDA error = %s from file <%s:%i>.\n",
            cudaGetErrorString(err), file, line
            );
        fflush(stderr);
        return false;
    }
    return true;
}

inline bool __checkLastCudaError(const char *file, const int line) {
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, 
            "[E] %s <%s:%i>\n",
            cudaGetErrorString(cudaGetLastError()), file, line
            );
        fflush(stderr);
        return false;
    }
    return true;
}



namespace mwe {
namespace device_util
{
    template<typename T> class device_ptr final
    {
        struct cuda_deleter final {
            void operator()(T* b) noexcept { checkCudaErrors(cudaFree(b)); }
        };

        std::unique_ptr<T, cuda_deleter> m_ptr;
        size_t m_size;

    public:
        device_ptr(size_t n)
            : m_ptr(nullptr), m_size(n)
        {
            T* p;
            checkCudaErrors(cudaMalloc((void**)&p, n * sizeof(T)));
            m_ptr.reset(p);
        }

        device_ptr()
            : device_ptr(1)
        {}

        device_ptr(device_ptr<T>&& other)
            : m_ptr(std::move(other.m_ptr)), m_size(other.m_size)
        {}

        device_ptr(gsl::span<T> span)
            : device_ptr(span.size())
        {
            copy_from(span);
        }

        gsl::span<T> span() {
            return { m_ptr.get(), m_size };
        }

        const gsl::span<T> span() const {
            return { m_ptr.get(), m_size };
        }

        T* get() {
            return m_ptr.get();
        }

        void copy_from(const gsl::span<T> from)
        {
            assert(m_size == from.size());
            checkCudaErrors(cudaMemcpy(
                m_ptr.get(), from.data(), from.size_bytes(),
                cudaMemcpyHostToDevice));
        }

        void copy_to(gsl::span<T> to) const
        {
            assert(m_size == to.size());
            checkCudaErrors(cudaMemcpy(
                to.data(), m_ptr.get(), to.size_bytes(),
                cudaMemcpyDeviceToHost));
        }
    };
}
}

#endif

namespace mwe {
namespace device_util
{
    bool init_cudart();
}
}