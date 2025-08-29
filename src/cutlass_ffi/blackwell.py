import jax
import jax.numpy as jnp
import numpy as np


def blackwell_group_gemm_block_scaled_sm100(
    a, b, asf, bsf, group_sizes, *, use_2sm: bool
):
    """fp4 blocked scaled group gemm.

    A/B are packed tensors consisting of multiple groups of activations and weights.
    ASF/BSF are the packed scale factors for A and B respectively.

    Group_Sizes is a vector containing the length of each group. The size of the
    vector indicates the number of groups.
    """
    la, m, ak = a.shape
    lb, n, bk = b.shape

    assert ak == bk, f"{ak=} does not match {bk=}"
    assert la == 1 and lb == 1, f"{la=} and {lb=} must be 1"

    num_groups = group_sizes.shape[0]
    assert n % num_groups == 0, f"{n=} must be multiple of {num_groups=}"
    n_per_group = n // num_groups

    # Build problem size array [..(m,k,l)..]
    n_sizes = jnp.full((num_groups,), n_per_group, jnp.int32)
    k_sizes = jnp.full((num_groups,), ak, jnp.int32)
    problem_shapes = jnp.concatenate(
        [group_sizes.reshape(-1, 1), n_sizes.reshape(-1, 1), k_sizes.reshape(-1, 1)],
        axis=-1,
    )

    # Compute cumulative offsets of groups
    group_offsets = jnp.cumsum(group_sizes) - group_sizes

    # There are two workspaces.
    # 1. Pointer workspace - holds all input pointers for each group
    # 2. Cutlass workspace - holds tensor maps for all inputs on each SM
    workspace_size, cutlass_workspace_size = (
        16 * 1024 * num_groups,
        113664 * 8 * num_groups,
    )
    call = jax.ffi.ffi_call(
        "blackwell_group_gemm_block_scaled_sm100",
        result_shape_dtypes=(
            jax.ShapeDtypeStruct((m, n_per_group), jnp.float16),
            jax.ShapeDtypeStruct((workspace_size,), dtype=jnp.uint8),
            jax.ShapeDtypeStruct((cutlass_workspace_size,), dtype=jnp.uint8),
        ),
    )
    d, workspace, cutlass_wrksapce = call(
        a, b, asf, bsf, problem_shapes, group_offsets, use_2sm=use_2sm
    )
    return d


def blackwell_group_gemm_block_scaled_sm103(
    a, b, asf, bsf, group_sizes, *, use_2sm: bool
):
    """fp4 ultra blocked scaled group gemm.

    A/B are packed tensors consisting of multiple groups of activations and weights.
    ASF/BSF are the packed scale factors for A and B respectively.

    Group_Sizes is a vector containing the length of each group. The size of the
    vector indicates the number of groups.
    """
    la, m, ak = a.shape
    lb, n, bk = b.shape

    assert ak == bk, f"{ak=} does not match {bk=}"
    assert la == 1 and lb == 1, f"{la=} and {lb=} must be 1"

    num_groups = group_sizes.shape[0]
    assert n % num_groups == 0, f"{n=} must be multiple of {num_groups=}"
    n_per_group = n // num_groups

    # Build problem size array [..(m,k,l)..]
    n_sizes = jnp.full((num_groups,), n_per_group, jnp.int32)
    k_sizes = jnp.full((num_groups,), ak, jnp.int32)
    problem_shapes = jnp.concatenate(
        [group_sizes.reshape(-1, 1), n_sizes.reshape(-1, 1), k_sizes.reshape(-1, 1)],
        axis=-1,
    )

    # Compute cumulative offsets of groups
    group_offsets = jnp.cumsum(group_sizes) - group_sizes

    # There are two workspaces.
    # 1. Pointer workspace - holds all input pointers for each group
    # 2. Cutlass workspace - holds tensor maps for all inputs on each SM
    workspace_size, cutlass_workspace_size = (
        16 * 1024 * num_groups,
        113664 * 8 * num_groups,
    )
    call = jax.ffi.ffi_call(
        "blackwell_group_gemm_block_scaled_sm103",
        result_shape_dtypes=(
            jax.ShapeDtypeStruct((m, n_per_group), jnp.float16),
            jax.ShapeDtypeStruct((workspace_size,), dtype=jnp.uint8),
            jax.ShapeDtypeStruct((cutlass_workspace_size,), dtype=jnp.uint8),
        ),
    )
    d, workspace, cutlass_wrksapce = call(
        a, b, asf, bsf, problem_shapes, group_offsets, use_2sm=use_2sm
    )
    return d
