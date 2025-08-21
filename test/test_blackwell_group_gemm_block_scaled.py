import pytest
from functools import partial
import random

import jax
import jax.numpy as jnp

import cutlass_ffi as cf
from tensor import (
    create_a_tensor,
    create_b_tensor,
    create_cd_tensor,
    gemm_a_mode,
    gemm_b_mode,
    gemm_cd_mode,
    gemm_reference_einsum,
)


@pytest.mark.parametrize(
    "problem_size",
    [
        pytest.param((4, 4 * 256, 256, 512), id="E4-M1024-N256-K512"),
        pytest.param((32, 32 * 1024, int(1.5 * 1024), 2048), id="E32-M32768-N1536-K2048"),
    ],
)
@pytest.mark.parametrize(
    "use_2sm",
    [
        pytest.param(False, id="1SM"),
        pytest.param(True, id="2SM"),
    ])
def test_blackwell_group_gemm_block_scaled(problem_size, use_2sm):
    def ceil_div(a, b):
        return (a + b - 1) // b

    key = jax.random.key(4281)

    ab_dtype = jnp.float4_e2m1fn
    sf_dtype = jnp.float8_e4m3
    d_dtype = jnp.float16

    a_major, b_major = "k", "k"
    d_major = "n"

    sf_vec_size = 16

    num_groups, m, n, k = problem_size
    sf_k = ceil_div(k, sf_vec_size)

    gkey, key = jax.random.split(key)
    group_sizes = jnp.array([m // num_groups] * num_groups)

    tensors_abd = []
    problem_sizes_mnkl = []
    strides_abd = []
    sfa_ref = []
    sfb_ref = []

    # Build separate tensors for each expert. It is expected that the total tokens will
    # sum to m. n is uniform across all experts.
    for idx in range(num_groups):
        sub_m = int(group_sizes[idx])
        akey, asfkey, bkey, bsfkey, dkey, key = jax.random.split(key, 6)

        tensor_a = create_a_tensor(1, sub_m, k, a_major, ab_dtype, akey, minval=-1.0, maxval=1.0)
        tensor_b = create_b_tensor(1, n, k, b_major, ab_dtype, bkey, minval=1.0, maxval=1.0)
        tensor_d = create_cd_tensor(1, sub_m, n, d_major, d_dtype, dkey, fill_value=0.0)

        # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        # Scale factors are using .scale_vec::4X / .block16 config to support nvfp4 and mxfp4
        atom_mn = (32, 4)
        atom_k = 4

        sfa = create_a_tensor(1, sub_m, sf_k, a_major, sf_dtype, asfkey, minval=1.0, maxval=3.0)
        sfa_ref.append(sfa)
        sfa = sfa.reshape(
            1,
            ceil_div(sub_m, atom_mn[0] * atom_mn[1]),
            atom_mn[1],
            atom_mn[0],
            ceil_div(sf_k, atom_k),
            atom_k,
        )
        sfa = sfa.transpose(0, 1, 4, 3, 2, 5)

        sfb = create_b_tensor(1, n, sf_k, b_major, sf_dtype, bsfkey, minval=1.0, maxval=1.0)
        sfb_ref.append(sfb)
        sfb = sfb.reshape(
            1,
            ceil_div(n, atom_mn[0] * atom_mn[1]),
            atom_mn[1],
            atom_mn[0],
            ceil_div(sf_k, atom_k),
            atom_k,
        )
        sfb = sfb.transpose(0, 1, 4, 3, 2, 5)

        tensors_abd.append((tensor_a, sfa, tensor_b, sfb, tensor_d))

    # Create the combined tensors by concatenating along the appropriate axis
    am_axis = gemm_a_mode(a_major)[0]  # mkl
    bn_axis = gemm_b_mode(b_major)[0]  # nkl
    dm_axis = gemm_cd_mode(d_major)[0]  # mnl
    tensor_a_device = jnp.concatenate([x[0] for x in tensors_abd], axis=am_axis)
    tensor_sfa_device = jnp.concatenate([x[1] for x in tensors_abd], axis=am_axis)
    tensor_b_device = jnp.concatenate([x[2] for x in tensors_abd], axis=bn_axis)
    tensor_sfb_device = jnp.concatenate([x[3] for x in tensors_abd], axis=bn_axis)
    tensor_d_device = jnp.concatenate([x[4] for x in tensors_abd], axis=dm_axis)

    # Call our kernel!
    gemm = jax.jit(cf.blackwell_group_gemm_block_scaled, static_argnames=["use_2sm"])
    tensor_d_device = gemm(
        tensor_a_device, tensor_b_device, tensor_sfa_device, tensor_sfb_device, group_sizes, use_2sm=use_2sm
    )

    d_ref = []
    for idx in range(num_groups):
        d_ref.append(
            gemm_reference_einsum(
                tensors_abd[idx][0],
                tensors_abd[idx][2],
                acc_dtype=jnp.float32,
                cd_dtype=jnp.float16,
                a_major=a_major,
                b_major=b_major,
                cd_major=d_major,
                sf_a=sfa_ref[idx],
                sf_b=sfb_ref[idx],
            )
        )
    d_ref = jax.numpy.squeeze(jnp.concatenate(d_ref, axis=dm_axis).astype(jnp.float32), 0)
    tensor_d_device = tensor_d_device.astype(jnp.float32)

    assert jnp.allclose(d_ref, tensor_d_device), "mismatch!"
