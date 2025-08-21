
#include "kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename T> pybind11::capsule EncapsulateFFI(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return pybind11::capsule(reinterpret_cast<void *>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["blackwell_group_gemm_block_scaled"] = EncapsulateFFI(BlackwellGroupGemmBlockScaledHandler);
  return dict;
}

PYBIND11_MODULE(cutlass_ffi_kernels, m) {
  m.def("registrations", &Registrations);
}
