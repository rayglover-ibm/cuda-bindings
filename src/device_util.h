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
        return false;
    }
    return true;
}

inline bool __checkLastCudaError(const char *file, const int line) {
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, 
            "[E] %s <%s:%i>",
            cudaGetErrorString(cudaGetLastError()), file, line
            );
        return false;
    }
    return true;
}

#endif

namespace cufoo {
namespace device_util
{
    bool init_cudart();

    template<typename T> class dev_ptr final
    {
        struct cuda_deleter final {
            void operator()(T* b) noexcept { checkCudaErrors(cudaFree(b)); }
        };

        std::unique_ptr<T, cuda_deleter> m_ptr;
        size_t m_size;

    public:
        dev_ptr(size_t n)
            : m_ptr(nullptr), m_size(n)
        {
            T* p;
            checkCudaErrors(cudaMalloc((void**)&p, n * sizeof(T)));
            m_ptr.reset(p);
        }

        dev_ptr(gsl::span<T> span) : dev_ptr(span.size()) {
            copy_from(span);
        }

        inline gsl::span<T> span() {
            return gsl::span<T>{ m_ptr.get(), m_size };
        }

        inline void copy_from(gsl::span<T> from)
        {
            if (from.size() > m_size) {
                fprintf(stderr, "Invalid size.");
                exit(-1);
            }
            checkCudaErrors(cudaMemcpy(
                m_ptr.get(), from.data(), from.size_bytes(), cudaMemcpyHostToDevice));
        }

        inline void copy_to(gsl::span<T> to)
        {
            if (to.size() > m_size) {
                fprintf(stderr, "Invalid size.");
                exit(-1);
            }
            checkCudaErrors(cudaMemcpy(
                to.data(), m_ptr.get(), to.size_bytes(), cudaMemcpyDeviceToHost));
        }
    };
}
}