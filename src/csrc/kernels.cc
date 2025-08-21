
#include "kernels.h"

#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <utility>

#include "cutlass/kernel_hardware_info.h"

template <typename T> pybind11::capsule EncapsulateFFI(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return pybind11::capsule(reinterpret_cast<void *>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["blackwell_group_gemm_block_scaled"] =
      EncapsulateFFI(BlackwellGroupGemmBlockScaledHandler);
  return dict;
}

static std::pair<std::unordered_map<int32_t, int32_t> *, std::mutex *>
GetDeviceSmCountCache() {
  static std::mutex *mutex = new std::mutex{};
  static std::unordered_map<int32_t, int32_t> *map =
      new std::unordered_map<int32_t, int32_t>{};
  return std::make_pair(map, mutex);
}

int32_t GetDeviceSmCount(int32_t device) {
  auto [cache, mutex] = GetDeviceSmCountCache();
  std::unique_lock<std::mutex> _lock{*mutex};
  if (cache->find(device) == cache->end()) {
    cache->insert(
        {device,
         cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0)});
  }
  return cache->at(device);
}

PYBIND11_MODULE(cutlass_ffi_kernels, m) {
  m.def("registrations", &Registrations);
  m.def("get_device_sm_count", &GetDeviceSmCount);
}
