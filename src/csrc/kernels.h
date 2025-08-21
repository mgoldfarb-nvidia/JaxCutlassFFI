#ifndef _CUTLASS_FFI_KERNELS_H_
#define _CUTLASS_FFI_KERNELS_H_

#include <cstdint>
#include <cuda_runtime_api.h>
#include <xla/ffi/api/ffi.h>

#include "cutlass/cutlass.h"

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

int32_t GetDeviceSmCount(int32_t device);

XLA_FFI_DECLARE_HANDLER_SYMBOL(BlackwellGroupGemmBlockScaledHandler);

#endif /*_CUTLASS_FFI_KERNELS_H_*/
