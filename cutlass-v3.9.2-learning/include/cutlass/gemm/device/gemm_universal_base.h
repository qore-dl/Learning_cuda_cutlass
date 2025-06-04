/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
  \file
  \brief The universal GEMM accommodates streamk, batched strided, and batched array variants.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/limits>
#else
#include <limits>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*
GemmUniversalBase 作为 GemmUniversal 和 GemmUniversalStreamk 中的基类，我们对其相关源码进行阅读。
给出相应地注释。
*/


template <typename GemmKernel_>
class GemmUniversalBase {
public:

  using GemmKernel = GemmKernel_;

  /// Boolean indicating whether the CudaHostAdapter is enabled
  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;

  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;

  /// Numerical accumulation element type
  using ElementAccumulator = typename GemmKernel::Mma::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Operator = typename GemmKernel::Operator;

  /// Argument structure
  /*
  Arguments使用传入GemmKernel结构体的类型。
  即 GemmUniversal::Arguments 或者GemmUniversalStreamk::Arguments。
  */
  using Arguments = typename GemmKernel::Arguments;


  /// Index of the GEMM Kernel within the CudaHostAdapter
  static int32_t const kGemmKernelIndex = 0;

  /// Kernel dynamic shared memory allocation requirement
  /// Update the kernel function's shared memory configuration for the current device
  static constexpr size_t kSharedStorageSize = sizeof(typename GemmKernel::SharedStorage);

protected:

  //
  // Device properties (uniform across all instances of the current thread)
  //

  // Device ordinal
  /*
  设备序号 device_ordinal_的初始值为-1。(见521行)记录了当前设备相关参数初始化已完成的设备的序号
  */
  CUTLASS_THREAD_LOCAL static int device_ordinal_;

  /// Device SM count
  CUTLASS_THREAD_LOCAL static int device_sms_;

  /// Kernel SM occupancy (in thread blocks)
  CUTLASS_THREAD_LOCAL static int sm_occupancy_;

protected:

  /// Initialize static thread-local members for the thread's current device,
  /// if necessary.
  /*
  在get_workspace_size，initialize 等函数中，均需解析arguments，因此我们需要调用init_params
  在此时，我们的初始化过程中，对设备参数的设置是一个关键组成部分，这要求我们使用：init_device_props函数
  */
  static Status init_device_props()
  {
    /*
    如果有必要，初始化线程当前设备的静态线程本地成员。
    CUTLASS_TRACE_HOST 在 debug 模式下，打印文件名和行号。
    */
    CUTLASS_TRACE_HOST("GemmUniversalBase::init_device_props()");

    cudaError_t cudart_result;

    // Get current device ordinal
    int current_ordinal;
    cudart_result = cudaGetDevice(&current_ordinal);
    if (cudart_result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  cudaGetDevice() returned error " << cudaGetErrorString(cudart_result));
      return Status::kErrorInternal;
    }

    // Done if matches the current static member
    /*
    cudaGetDevice 返回当前正在使用的设备。
    如果当前设备已经初始化了，则直接返回。
    */
    if (current_ordinal == device_ordinal_) {
      // Already initialized
      return Status::kSuccess;
    }

    // Update SM count member
    /*
    cudaDeviceGetAttribute 返回有关设备的信息。
    此时，初始化device_sms_，获取设备的SM的数量
    */
    cudart_result = cudaDeviceGetAttribute (&device_sms_, cudaDevAttrMultiProcessorCount, current_ordinal);
    if (cudart_result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  cudaDeviceGetAttribute() returned error " << cudaGetErrorString(cudart_result));
      return Status::kErrorInternal;
    }

    // If requires more than 48KB: configure for extended, dynamic shared memory
    if constexpr (kSharedStorageSize >= (48 << 10))
    {
      /*
      cudaFuncSetAttribute 设置给定函数的属性：
      cudaError_t cudaFuncSetAttribute(
        const void *func,
        cudaFuncAttribute attr,
        int value
      );
      func: 内核函数的指针。这个参数指定了要设置属性的 CUDA 内核函数。
      attr: 要设置的属性类型。这个参数是一个枚举类型 cudaFuncAttribute，用于指定要设置的具体属性。
      value: 属性的值。这个参数指定了要为 attr 设置的具体值。

      此处意味着，如果设备支持的 SharedMemory 大于48KB，
      则更新 GemmKernel在SM 上动态分配的共享内存的最大容量的设置。
      */
      cudart_result = cudaFuncSetAttribute(
        Kernel2<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedStorageSize);
      if (cudart_result != cudaSuccess) {
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error " << cudaGetErrorString(cudart_result));
        return Status::kErrorInternal;
      }
    }

    // Update SM occupancy member
    /*
    设置 sm_occupancy_：cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags 是 CUDA Runtime API，
    返回每个 SM 运行该 kernel 函数（Kernel2<GemmKernel>）时可以支持的最大活跃线程块数。

    具体来说，cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags 用于计算每个多处理器（SM）上可以同时活动的
    最大线程块数。这个函数考虑了特定的标志（flags），以便更准确地反映某些内核配置选项对占用率的影响。
    函数原型如下：
    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      int *numBlocks,
      const void *func,
      int blockSize,
      size_t dynamicSMemSize,
      unsigned int flags
    );
    参数说明：
    numBlocks: 指向一个整数的指针，用于存储计算得到的每个多处理器上可以同时活动的最大线程块数。
    func: 内核函数的指针。指定要计算占用率的 CUDA 内核。
    blockSize: 每个线程块中的线程数。指定内核的线程块大小。
    dynamicSMemSize: 每个线程块动态分配的共享内存大小（以字节为单位）。
    flags: 用于指定影响内核占用率计算的标志。可以用来调整计算方式或考虑特定的内核配置选项。
    
    返回值：
    cudaSuccess: 如果函数成功执行，返回 cudaSuccess。
    如果函数执行失败，返回相应的错误代码，开发者可以使用 cudaGetErrorString 获取错误信息。
    */
    cudart_result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &sm_occupancy_,
      Kernel2<GemmKernel>,
      GemmKernel::kThreadCount,
      kSharedStorageSize,
      cudaOccupancyDisableCachingOverride);
    if (cudart_result != cudaSuccess) {
      CUTLASS_TRACE_HOST("  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() returned error " << cudaGetErrorString(cudart_result));
      return Status::kErrorInternal;
    }

    // Update device ordinal member on success
    device_ordinal_ = current_ordinal;

    CUTLASS_TRACE_HOST("  "
      "device_ordinal: (" << device_ordinal_ << "), "
      "device_sms: (" << device_sms_ << "), "
      "sm_occupancy: (" << sm_occupancy_ << ") "
      "smem_size: (" << kSharedStorageSize << ") "
      "GemmKernel::kThreadCount: (" << GemmKernel::kThreadCount << ")");

    return Status::kSuccess;
  }


protected:

  //
  // Instance data members
  //

  /// Kernel parameters

  /*
  此处为GemmKernel 中的参数，因为kernel的具体类型不同，
  可能是 GemmUniversal::Params 或者 GemmUniversalStreamk::Params。
  具体可见：
  include/cutlass/gemm/kernel/gemm_universal.h (GemmUniversal)
  include/cutlass/gemm/kernel/gemm_universal_streamk.h (GemmUniversalStreamk)
  */
  typename GemmKernel::Params params_;


  /// Initialize params member
  /*
  在熟悉了适应不同硬件配置下kernel属性设置的init_device_props函数的基础上，我们解析参数初始化函数 init_params
  */
  Status init_params(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    int32_t device_sms = 0;
    int32_t sm_occupancy = 0;

    // kEnableCudaHostAdapter 的值为宏CUTLASS_ENABLE_CUDA_HOST_ADAPTER，未启用。(79 行)
    // CudaHostAdapter 类也没有实现。
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);

      //
      // Occupancy query using CudaHostAdapter::query_occupancy().
      //

      if (cuda_adapter) {
        /*
        若实现了cuda 适配器
        */
        Status status = cuda_adapter->query_occupancy(
          &device_sms,
          &sm_occupancy,
          kGemmKernelIndex,
          GemmKernel::kThreadCount,
          kSharedStorageSize);

        CUTLASS_ASSERT(status == Status::kSuccess);

        if (status != Status::kSuccess) {
          return status;
        }
      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      /*
      若未指明 cuda 适配器，调用init_device_props函数得到 SM 数量和 SM 内的最大线程块数。
      */
      CUTLASS_ASSERT(cuda_adapter == nullptr);

      // Initialize static device properties, if necessary
      Status result = init_device_props();

      if (result != Status::kSuccess) {
        return result;
      }

      //
      // Use thread-local static members for occupancy query initialized by call to
      // `init_device_props()`
      //

      device_sms   = device_sms_;
      sm_occupancy = sm_occupancy_;
    }

    // Initialize params member
    /*
    得到一个 GemmUniversal::Params 或者 GemmUniversalStreamk::Params 对象。
    */
    params_ = typename GemmKernel::Params(args, device_sms, sm_occupancy);
    return Status::kSuccess;
  }

public:

  //---------------------------------------------------------------------------------------------
  // Stateless API
  //---------------------------------------------------------------------------------------------

  /// Determines whether the GEMM can execute the given problem.
  /*
  GemmUniversalBase::can_implement 判断能否实现： grid 是否超出以及形状是否满足对齐要求。
  调用 kernel 的 GemmUniversal::can_implement 或 GemmUniversalStreamk::can_implement 进一步检查。
  */
  static Status can_implement(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::can_implement()");

    if (!kEnableCudaHostAdapter || cuda_adapter) {
      /*
      获取网格规模，判断GPU是否可以承载网格规模
      */
      dim3 grid = get_grid_shape(args, cuda_adapter);

      if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
            grid.z <= std::numeric_limits<uint16_t>::max()))
      {
        return Status::kErrorInvalidProblem;
      }
    }
    else {
      //
      // With a null host adapter, a conservative grid shape is computed and required to conform to CUDA grid
      // dimension limits.
      //

      int64_t logicalGridM = (int64_t(args.problem_size.m()) + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
      int64_t logicalGridN = (int64_t(args.problem_size.n()) + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
      int32_t logicalGridL = args.batch_count;

      if ((int64_t(std::numeric_limits<uint32_t>::max()) < logicalGridM) ||
          (int64_t(std::numeric_limits<uint16_t>::max()) < logicalGridN) ||
          (int32_t(std::numeric_limits<uint16_t>::max()) < logicalGridL)) {

        return Status::kErrorInvalidProblem;
      }

    }
    /*
    调用 kernel 的 GemmUniversal::can_implement 或 GemmUniversalStreamk::can_implement 进一步检查。
    */
    return GemmKernel::can_implement(args);
  }


  /// Returns the workspace size (in bytes) needed for the problem
  /// geometry expressed by these arguments
  static size_t get_workspace_size(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    /*
    返回由这些参数表示的问题几何形状所需的工作区大小（以字节为单位）。
    */
    CUTLASS_TRACE_HOST("GemmUniversalBase::get_workspace_size()");

    // Initialize parameters from args
    /*
    首先创建一个 GemmUniversalBase 对象。
    */
    GemmUniversalBase base;
    /*
    然后调用 GemmUniversalBase::init_params 初始化参数。
    */
    if (base.init_params(args, cuda_adapter) != Status::kSuccess) {
      return 0;
    }

    // Get size from parameters

    size_t workspace_bytes = base.params_.get_workspace_size();

    /*
    实质上是通过GemmKernel 中的具体workspace计算，
    例如，调用 UniversalParamsBase::get_workspace_size 或者 
    GemmUniversalStreamk::Params::get_workspace_size 函数得到 kernel 需要的全局内存工作空间大小。
    */

    /*
    例如，对于kernel::GemmUniversalBase 而言，在data-parallel 和split-k情况下估算如下：
    /// Returns the workspace size (in bytes) needed for this problem geometry
    size_t get_workspace_size() const
    {
    size_t workspace_bytes = 0;
    if (mode == GemmUniversalMode::kGemmSplitKParallel)
    {
      // Split-K parallel always requires a temporary workspace
      workspace_bytes =
        sizeof(ElementC) *
        size_t(batch_stride_D) *
        size_t(grid_tiled_shape.k());
    }
    else if (mode == GemmUniversalMode::kGemm && grid_tiled_shape.k() > 1)
    {
      // Serial split-K only requires a temporary workspace if the number of partitions along the
      // GEMM K dimension is greater than one.
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    return workspace_bytes;
    }
    在 stream-k的情况下，需要：1. Split-K 时临时存储partial 计算的结果 + 进行 partial 之间的reduction的结果，需要相应的存储空间：
    /// Returns the workspace size (in bytes) needed for these parameters
    size_t get_workspace_size() const
    {
      return
        get_barrier_workspace_size() +
        get_partials_workspace_size();
    }

    /// Get the workspace size needed for barrier
    size_t get_barrier_workspace_size() const
    {
      // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
      // each reduction block needs its own synchronization flag.
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

      return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
    }

    /// Get the workspace size needed for intermediate partial sums
    size_t get_partials_workspace_size() const
    {
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
    }
    */

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);
    return workspace_bytes;
  }


  /// Returns the grid extents in thread blocks to launch
  static dim3 get_grid_shape(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::get_grid_shape()");

    // Initialize parameters from args
    /*
    首先创建一个 GemmUniversalBase 对象。
    */
    GemmUniversalBase base;
    /*
    然后调用 GemmUniversalBase::init_params 初始化参数。
    */
    if (base.init_params(args, cuda_adapter) != Status::kSuccess) {
      return dim3(0,0,0);
    }

    // Get dims from parameters
    /*
    调用GemmKernel具体的grid维度获取方式，得到网格维度：
    例如：
    GemmUniversalStreamk::Params::get_grid_dims 函数：
    // Initialize the block mapping structure
      block_mapping = ThreadblockSwizzle( // block 在 SM上或者grid上调度的方式
        args.mode,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.batch_count,
        sm_occupancy,
        device_sms,
        avail_sms,
        sizeof(ElementA),
        sizeof(ElementB),
        sizeof(ElementC),
        Epilogue::kAccumulatorFragments);
    }
    /// Returns the grid extents in thread blocks to launch
    dim3 get_grid_dims() const
    {
      return block_mapping.get_grid_dims();
    }
    

    
    */
    dim3 grid_dims = base.params_.get_grid_dims();

    CUTLASS_TRACE_HOST(
         "  tiled_shape: " << base.params_.get_tiled_shape()  << "\n"
      << "  grid_dims: {" << grid_dims << "}");

    return grid_dims;
  }


  /// Returns the maximum number of active thread blocks per multiprocessor'
  /*
  与 GemmUniversalBase::init_params 中的操作类似。
  利用 init_device_props() 计算一个 SM 中可以支持的最大active block 的数量
  */
  static int maximum_active_blocks(CudaHostAdapter *cuda_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

    int32_t device_sms   = 0;
    int32_t sm_occupancy = 0;


    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);

      if (cuda_adapter) {

        Status status = cuda_adapter->query_occupancy(
          &device_sms,
          &sm_occupancy,
          kGemmKernelIndex,
          GemmKernel::kThreadCount,
          kSharedStorageSize);

        CUTLASS_ASSERT(status == Status::kSuccess);

        if (status != Status::kSuccess) {
        return -1;
        }
      }
      else {
        return -1;
      }
    }
    else {
      CUTLASS_ASSERT(cuda_adapter == nullptr);
      // Initialize static device properties, if necessary
      if (init_device_props() != Status::kSuccess) {
        return -1;
      }

      sm_occupancy = sm_occupancy_;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << sm_occupancy_);
    return sm_occupancy;
  }


  //---------------------------------------------------------------------------------------------
  // Stateful API
  //---------------------------------------------------------------------------------------------

  /// Initializes GEMM state from arguments and workspace memory
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize parameters from args
    Status result = init_params(args, cuda_adapter);
    if (result != Status::kSuccess) {
      return result;
    }

    // Assign and prepare workspace memory
    if (args.mode == GemmUniversalMode::kGemm) {
      return params_.init_workspace(workspace, stream);
    }

    return Status::kSuccess;
  }


  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase()::update()");
    params_.update(args);
    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::run()");

    // Configure grid and block dimensions
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    dim3 grid = params_.get_grid_dims();

    // Launch kernel
    CUTLASS_TRACE_HOST("  "
      "grid: (" << grid << "), "
      "block: (" << block << "), "
      "SMEM: (" << kSharedStorageSize << ")");

    cutlass::arch::synclog_setup();

    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      if (cuda_adapter) {
        void* kernel_params[] = {&params_};
        return cuda_adapter->launch(grid, block, kSharedStorageSize, stream, kernel_params, 0);
      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      CUTLASS_ASSERT(cuda_adapter == nullptr);

      Kernel2<GemmKernel><<<grid, block, kSharedStorageSize, stream>>>(params_);

      // Query for errors
      cudaError_t result = cudaGetLastError();
      if (result != cudaSuccess) {
        CUTLASS_TRACE_HOST("  grid launch failed with error " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }


  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr)
  {
    return run(stream, cuda_adapter);
  }


  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr)
  {
    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter);
    }

    return status;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Static initializers
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Device ordinal
/*
device_ordinal_的初始值为-1。
*/
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::device_ordinal_ = -1;

/// Device SM count
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::device_sms_ = -1;

/// Kernel SM occupancy (in thread blocks)
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::sm_occupancy_ = -1;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
