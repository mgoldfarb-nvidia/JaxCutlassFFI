import jax

import cutlass_ffi_kernels

from .blackwell import *

for _name, _target in cutlass_ffi_kernels.registrations().items():
    jax.ffi.register_ffi_target(_name, _target, platform="CUDA")
