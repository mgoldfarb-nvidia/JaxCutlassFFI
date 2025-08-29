/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <xla/ffi/api/ffi.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "kernels.h"
#include "util.h"

/// CUTLASS DEFINITIONS ///

// From
// https://github.com/NVIDIA/cutlass/blob/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu

using namespace cute;

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group
using ElementInput =
    cutlass::float_e2m1_t;  // Element type for Input matrix operands
using ElementSF = cutlass::float_e4m3_t;  // Element type for SF matrix operands
using ElementC = cutlass::half_t;         // Element type for C matrix operands
using ElementD = cutlass::half_t;         // Element type for D matrix operands

// A matrix configuration
using ElementA =
    cutlass::nv_float4_t<ElementInput>;     // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    32;  // Alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB =
    cutlass::nv_float4_t<ElementInput>;  // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    32;  // Alignment of A matrix in units of elements (up to 16 bytes)

constexpr int InputSFVectorSize = 16;

// C/D matrix configuration
using LayoutC =
    cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
using LayoutD = LayoutC;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD =
    128 /
    cutlass::sizeof_bits<ElementD>::value;  // Alignment of D matrix in units of
                                            // elements (up to 16 bytes)
using ElementAccumulator = float;  // Element type for internal accumulation

// using ElementD = cutlass::float_e2m1_t; // Enable for SF Output          //
// Element type for D matrix operands

using ElementSFD =
    cutlass::float_ue4m3_t;  // Element type for SF Output operands
constexpr int OutputSFVectorSize = 16;
using FusionOperation =
    cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
        cutlass::epilogue::thread::SiLu, OutputSFVectorSize, ElementD,
        ElementAccumulator, ElementSFD, LayoutC, ElementC>;

// Core kernel configurations
using ArchTag = cutlass::arch::Sm100;  // Tag indicating the minimum SM that
                                       // supports the intended feature
using EpilogueOperatorClass =
    cutlass::arch::OpClassTensorOp;  // Epilogue Operator class tag
using MainloopOperatorClass =
    cutlass::arch::OpClassBlockScaledTensorOp;  // Mainloop Operator class tag
using StageCountType =
    cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based
                                                // on the tile size

// Runtime Cluster Shape
using ClusterShape = Shape<int32_t, int32_t, _1>;

// Different configs for 1SM and 2SM MMA kernel
struct MMA1SMConfig {
  using MmaTileShape = Shape<_128, _256, _256>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;  // Kernel to
                                                                    // launch
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;  // Epilogue to launch
};

struct MMA2SMConfig {
  using MmaTileShape = Shape<_256, _256, _256>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100;  // Kernel to
                                                                    // launch
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
};

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, EpilogueOperatorClass, typename MMA1SMConfig::MmaTileShape,
        ClusterShape, Shape<_128, _64>, ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
        typename MMA1SMConfig::EpilogueSchedule
        // , FusionOperation  // Enable for SF Output
        >::CollectiveOp;
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, MainloopOperatorClass, ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB, ElementAccumulator,
        typename MMA1SMConfig::MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        typename MMA1SMConfig::KernelSchedule>::CollectiveOp;
using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                         CollectiveEpilogue>;
using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Gemm = Gemm1SM;

using CollectiveEpilogue2SM =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, EpilogueOperatorClass, typename MMA2SMConfig::MmaTileShape,
        ClusterShape, Shape<_128, _64>, ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
        typename MMA2SMConfig::EpilogueSchedule
        // , FusionOperation  // Enable for SF Output
        >::CollectiveOp;
using CollectiveMainloop2SM =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, MainloopOperatorClass, ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB, ElementAccumulator,
        typename MMA2SMConfig::MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        typename MMA2SMConfig::KernelSchedule>::CollectiveOp;
using GemmKernel2SM =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                         CollectiveEpilogue2SM>;
using Gemm2SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA =
    typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB =
    typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using Sm1xxBlockScaledOutputConfig =
    cutlass::detail::Sm1xxBlockScaledOutputConfig<
        OutputSFVectorSize,
        cute::is_same_v<typename FusionOperation::GmemLayoutTagScalefactor,
                        cutlass::layout::RowMajor>
            ? cute::UMMA::Major::K
            : cute::UMMA::Major::MN>;
using OutputSFAtom = typename Sm1xxBlockScaledOutputConfig::SfAtom;
using LayoutSFD = typename Sm1xxBlockScaledOutputConfig::LayoutSF;

using GemmElementA = typename Gemm::ElementA;
using GemmElementB = typename Gemm::ElementB;
using GemmElementD = typename Gemm::ElementD;
using GemmElementSF = typename Gemm::GemmKernel::ElementSF;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

/// FFI Call

template <typename Gemm>
static xla::ffi::Error BlackwellGroupGemmBlockScaledImpl(
    cudaStream_t stream, int32_t device, const GemmElementA** A,
    const GemmElementB** B, const GemmElementSF** ASF,
    const GemmElementSF** BSF, UnderlyingProblemShape* Problem_Sizes,
    int32_t Num_Groups, StrideA* Stride_A, StrideB* Stride_B, StrideD* Stride_D,
    LayoutSFA* SFA_Layout, LayoutSFB* SFB_Layout, GemmElementD** D,
    uint8_t* cutlass_workspace) {
  typename Gemm::Arguments arguments;

  // Only single value for alpha and beta
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = GetDeviceSmCount(device);

  hw_info.cluster_shape = dim3(2, 1, 1);
  hw_info.cluster_shape_fallback = dim3(2, 1, 1);

  if (size<0>(
          typename Gemm::GemmKernel::CollectiveMainloop::AtomThrShapeMNK{}) ==
          2 &&
      (hw_info.cluster_shape.x < 2 || hw_info.cluster_shape_fallback.x < 2)) {
    std::string err =
        "Error: MMA2SMConfig kernel config needs cluster_dim.x >= 2";
    return xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL, err);
  }

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order =
      cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {
          Num_Groups, Problem_Sizes, nullptr /*host size*/
      },
      {A, Stride_A, B, Stride_B, ASF, SFA_Layout, BSF, SFB_Layout},
      {fusion_args, nullptr /*C*/, nullptr /*strideC*/, D, Stride_D},
      hw_info,
      scheduler};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, cutlass_workspace));
  CUTLASS_CHECK(gemm.run(stream, /* cuda_adapter = */ nullptr,
                         /* launch_with_pdl = */ false));

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return xla::ffi::Error(
        XLA_FFI_Error_Code_INTERNAL,
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return xla::ffi::Error::Success();
}

__global__ void BlackwellGroupGemmBlockScaled_PrepareWorkspace(
    const GemmElementA* A, const GemmElementB* B, const GemmElementSF* ASF,
    const GemmElementSF* BSF, GemmElementD* D,
    const UnderlyingProblemShape* problem_sizes, const int32_t* group_offsets,
    int32_t num_groups, const GemmElementA** Aptrs, const GemmElementB** Bptrs,
    const GemmElementSF** ASFptrs, const GemmElementSF** BSFptrs,
    GemmElementD** Dptrs, StrideA* Strides_A, StrideB* Strides_B,
    StrideD* Strides_D, LayoutSFA* Layouts_SFA, LayoutSFB* Layouts_SFB) {
  // Extract the pointer for the start of each sub-matrix in the packed A, B and
  // D Strides and layout also computed here.
  unsigned group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx < num_groups) {
    const auto problem_size = problem_sizes[group_idx];  // (m, n, k)
    const int32_t M = cute::get<0>(problem_size);
    const int32_t N = cute::get<1>(problem_size);
    const int32_t K = cute::get<2>(problem_size);
    const int32_t offset = group_offsets[group_idx];
    Aptrs[group_idx] = A + offset * K / 2;         // div by 2 because sub byte!
    Bptrs[group_idx] = B + group_idx * K * N / 2;  // div by 2 because sub byte!
    Dptrs[group_idx] = D + offset * N;

    ASFptrs[group_idx] = ASF + offset * K / InputSFVectorSize;
    BSFptrs[group_idx] = BSF + group_idx * N * K / InputSFVectorSize;

#if 0
    cute::print("group_idx = %d, offset = %d M=%d, N=%d, K=%d Aptr=%p Bptr=%p, Dptr=%p ASFptr=%p BSFptr=%p\n",
                group_idx, offset, M, N, K, Aptrs[group_idx], Bptrs[group_idx], Dptrs[group_idx], ASFptrs[group_idx], BSFptrs[group_idx]);
#endif

    // (512,_1,_0)
    auto sA = StrideA{};
    cute::get<0>(sA) = static_cast<long>(K);
    Strides_A[group_idx] = sA;

    auto sB = StrideB{};
    cute::get<0>(sB) = static_cast<long>(K);
    Strides_B[group_idx] = sB;

    auto sD = StrideD{};
    cute::get<0>(sD) = static_cast<long>(N);
    Strides_D[group_idx] = sD;

    // (((_32,_4),2),((_16,_4),8),(_1,1)):(((_16,_4),4096),((_0,_1),_512),(_0,8192))
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    Layouts_SFA[group_idx] = layout_SFA;

    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));
    Layouts_SFB[group_idx] = layout_SFB;
  }
}

xla::ffi::Error BlackwellGroupGemmBlockScaledSm100(
    cudaStream_t stream, int32_t device, xla::ffi::AnyBuffer A,
    xla::ffi::AnyBuffer B, xla::ffi::AnyBuffer A_scales,
    xla::ffi::AnyBuffer B_scales, xla::ffi::AnyBuffer Problem_Sizes,
    xla::ffi::AnyBuffer Group_Offsets, xla::ffi::Result<xla::ffi::AnyBuffer> D,
    xla::ffi::Result<xla::ffi::AnyBuffer> Workspace,
    xla::ffi::Result<xla::ffi::AnyBuffer> CutlassWorkspace, bool use_2sm) {
  const int32_t num_groups = Problem_Sizes.dimensions()[0];

  WorkspaceBuffer wkrspc(reinterpret_cast<uint8_t*>(Workspace->untyped_data()),
                         Workspace->dimensions()[0]);

  const GemmElementA** Aptrs = wkrspc.allocate<const GemmElementA*>(num_groups);
  const GemmElementB** Bptrs = wkrspc.allocate<const GemmElementB*>(num_groups);
  const GemmElementSF** SFAptrs =
      wkrspc.allocate<const GemmElementSF*>(num_groups);
  const GemmElementSF** SFBptrs =
      wkrspc.allocate<const GemmElementSF*>(num_groups);
  GemmElementD** Dptrs = wkrspc.allocate<GemmElementD*>(num_groups);

  StrideA* Stride_A_ = wkrspc.allocate<StrideA>(num_groups);
  StrideB* Stride_B_ = wkrspc.allocate<StrideB>(num_groups);
  StrideD* Stride_D_ = wkrspc.allocate<StrideD>(num_groups);

  LayoutSFA* SFA_Layout_ = wkrspc.allocate<LayoutSFA>(num_groups);
  LayoutSFB* SFB_Layout_ = wkrspc.allocate<LayoutSFB>(num_groups);

  if (Aptrs == nullptr || Bptrs == nullptr || SFAptrs == nullptr ||
      SFBptrs == nullptr || Dptrs == nullptr || Stride_A_ == nullptr ||
      Stride_B_ == nullptr || Stride_D_ == nullptr || SFA_Layout_ == nullptr ||
      SFB_Layout_ == nullptr) {
    return xla::ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                           "Workspace buffer allocation is too small.");
  }

  uint8_t* cutlass_workspace =
      reinterpret_cast<uint8_t*>(CutlassWorkspace->untyped_data());
  UnderlyingProblemShape* Problem_Shape_ =
      reinterpret_cast<UnderlyingProblemShape*>(Problem_Sizes.untyped_data());

  // Fill out the workspace for this kernel. Each thread processes one of the
  // problem blocks.
  const int32_t threads_per_block = 256;
  const int32_t nblocks =
      (num_groups + threads_per_block - 1) / threads_per_block;
  BlackwellGroupGemmBlockScaled_PrepareWorkspace<<<nblocks, threads_per_block,
                                                   0, stream>>>(
      reinterpret_cast<const GemmElementA*>(A.untyped_data()),
      reinterpret_cast<const GemmElementB*>(B.untyped_data()),
      reinterpret_cast<const GemmElementSF*>(A_scales.untyped_data()),
      reinterpret_cast<const GemmElementSF*>(B_scales.untyped_data()),
      reinterpret_cast<GemmElementD*>(D->untyped_data()), Problem_Shape_,
      reinterpret_cast<const int32_t*>(Group_Offsets.untyped_data()),
      num_groups, Aptrs, Bptrs, SFAptrs, SFBptrs, Dptrs, Stride_A_, Stride_B_,
      Stride_D_, SFA_Layout_, SFB_Layout_);

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return xla::ffi::Error(
        XLA_FFI_Error_Code_INTERNAL,
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }

  if (use_2sm) {
    return BlackwellGroupGemmBlockScaledImpl<Gemm2SM>(
        stream, device, Aptrs, Bptrs, SFAptrs, SFBptrs, Problem_Shape_,
        num_groups, Stride_A_, Stride_B_, Stride_D_, SFA_Layout_, SFB_Layout_,
        Dptrs, cutlass_workspace);
  } else {
    return BlackwellGroupGemmBlockScaledImpl<Gemm1SM>(
        stream, device, Aptrs, Bptrs, SFAptrs, SFBptrs, Problem_Shape_,
        num_groups, Stride_A_, Stride_B_, Stride_D_, SFA_Layout_, SFB_Layout_,
        Dptrs, cutlass_workspace);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BlackwellGroupGemmBlockScaledSm100Handler,
    BlackwellGroupGemmBlockScaledSm100,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::DeviceOrdinal>()
        .Arg<xla::ffi::AnyBuffer>()  // A fp4
        .Arg<xla::ffi::AnyBuffer>()  // B fp4
        .Arg<xla::ffi::AnyBuffer>()  // A_scales fp8
        .Arg<xla::ffi::AnyBuffer>()  // B_scales fp8
        .Arg<xla::ffi::AnyBuffer>()  // Group_Offsets s32[G+1]
        .Arg<xla::ffi::AnyBuffer>()  // Problem_Sizes s32[G][3]
        .Ret<xla::ffi::AnyBuffer>()  // D f16
        .Ret<xla::ffi::AnyBuffer>()  // Workspace buffer
        .Ret<xla::ffi::AnyBuffer>()  // Cutlass worksapce buffer
        .Attr<bool>("use_2sm"),
    {xla::ffi::Traits::kCmdBufferCompatible});
