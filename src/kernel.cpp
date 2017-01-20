#include "kernel.h"

namespace cufoo {
namespace kernel
{
    const char* to_str(const status s) {
        switch (s) {
            case status::KERNEL_FAILED: return "Kernel Failed";
            case status::KERNEL_UNAVILABLE: return "Kernel Unavilable";
            case status::KERNEL_NOT_DEFINED: return "Kernel Not Defined";
            case status::CANCELLED: return "Cancelled";
            case status::SUCCESS: return "Success";
        }
        return "Unknown";
    }

    const char* to_str(const compute_mode m) {
        switch (m) {
            case compute_mode::CPU: return "CPU";
            case compute_mode::CUDA: return "Cuda";
            case compute_mode::AUTO: return "Auto";
        }
        return "Unknown";
    }

    compute_mode detail::runtime_mode()
    {
        if (compute_traits<compute_mode::CUDA>::enabled)
            return compute_traits<compute_mode::CUDA>::available() ?
                compute_mode::CUDA : compute_mode::CPU;
        else
            return compute_mode::CPU;
    }
}
}