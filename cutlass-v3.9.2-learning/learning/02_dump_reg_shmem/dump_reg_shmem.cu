
/*
Demonstrate CUTLASS debugging tool for dumping fragments and shared memory
演示用于转储段和共享内存的cutlass debugging 工具
*/

// Standard Library includes
#include <iostream>

//
// CUTLASS includes
//
#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#define EXAMPLE_MATRIX_ROW 64
#define EXAMPLE_MATRIX_COL 32

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element,typename GmemIterator,typename SmemIterator>
__global__ void kernel_dump(typename GmemIterator::Params params,
                            typename GmemIterator::TensorRef ref){

    extern __shared__ Element shared_storage[];
    // Construct the global iterator and load the data to the fragments.
    int tb_thread_id = threadIdx.y*blockDim.x + threadIdx.x;

    GmemIterator gmem_iterator(params,ref.data(),
                               {EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL},
                               tb_thread_id);

    /*
    在 NVIDIA 的 Cutlass 库中，GmemIterator::Fragment 是一种用于表示全局内存（global memory）迭代器片段的类型。为了更好地理解其作用，我们可以分解这个概念：
    GmemIterator:
    GmemIterator 是一种迭代器类型，用于在全局内存中读取或写入数据。全局内存是 GPU 上的一种内存类型，具有较大的容量，但相对于共享内存来说，访问延迟较高。
    这种迭代器通常用于从全局内存中获取数据到片段（fragment），或者将数据从片段写回到全局内存。
    
    Fragment:
    Fragment 是 Cutlass 中的一个核心概念，用于表示一小块矩阵或张量数据。片段通常被设计为适合在寄存器或共享内存中存储，以便于快速访问和计算。
    在上下文中，GmemIterator::Fragment 表示一个片段，专门用于存储从全局内存中迭代读取的数据块。
    
    作用:
    GmemIterator::Fragment frag; 声明了一个片段对象 frag，它将用于存储通过 GmemIterator 从全局内存中读取的数据。
    在使用 Cutlass 进行矩阵运算时，计算通常是基于这些片段进行的，因为它们可以有效地映射到 GPU 的计算资源。
    */

    typename GmemIterator::Fragment frag;

    frag.clear();
    gmem_iterator.load(frag);

    // Call dump_fragment() with different parameters.
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("\nAll threads dump all the elements:\n");
    }
    /*
    frag: frag 是一个片段对象，通常代表一个小块的矩阵或张量数据。片段在 Cutlass 中用于在 GPU 上执行分块计算，以便更好地利用硬件资源。
    cutlass::debug::dump_fragment:这是一个调试函数，用于将片段 frag 的内容打印到标准输出或日志中。
    它可以帮助开发者在调试和验证算法时，检查片段中的数据是否正确，或者是否符合预期。
    这种调试输出对于理解和优化复杂的 GPU 内核计算非常有用，因为它允许开发者查看计算过程中每个步骤的中间结果。
    在使用这个函数时，需要确保编译时启用了调试支持，因为在某些情况下，调试功能可能在发布版本中被禁用以优化性能。
    通过这种调试工具，开发者可以更好地理解程序的行为，尤其是在开发和优化 GPU 加速的数值计算时。
    */
    cutlass::debug::dump_fragment(frag);

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("\nFirst thread dumps all the elements:\n");
    }
    cutlass::debug::dump_fragment(frag, /*N = */ 1);

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("\nFirst thread dumps first 16 elements:\n");
    }
    cutlass::debug::dump_fragment(frag,/*N = */1,/*M = */16);
    
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
    }
    cutlass::debug::dump_fragment(frag,/*N = */1,/*M = */16,/*S = */8);

    // Construct the shared iterator and store the data to the shared memory.
    SmemIterator smem_iterator(
        typename SmemIterator::TensorRef(
            {shared_storage,SmemIterator::Layout::packed(
                {EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL})}),
        tb_thread_id);
    
    smem_iterator.store(frag);

    // Call dump_shmem() with different parameters.
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("\nDump all the elements:\n");
    cutlass::debug::dump_shmem(shared_storage,EXAMPLE_MATRIX_ROW*EXAMPLE_MATRIX_COL);

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("\nDump all the elements with a stride of 8:\n");
    }

    cutlass::debug::dump_shmem(shared_storage,EXAMPLE_MATRIX_ROW*EXAMPLE_MATRIX_COL,/*S = */8);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point for dump_reg_shmem example.
//
// usage:
//
//   02_dump_reg_shmem
//

int main(){
    // Initialize a 64x32 column major matrix with sequential data (1,2,3...).
    using Element = cutlass::half_t;
    using Layout = cutlass::layout::ColumnMajor;

    cutlass::HostTensor<Element,Layout> matrix({EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL});
    cutlass::reference::host::BlockFillSequential(matrix.host_data(),matrix.capacity());

    // Dump the Matrix:
    std::cout << "Matrix:\n" << matrix.host_view() << "\n";

    // Copy the matrix to the device.
    matrix.sync_device();

    // Define a global iterator, a shared iterator and their thread map.
    /*
    ThreadMap 是一种线程映射策略，用于将计算线程映射到矩阵块上。
    PitchLinearWarpRakedThreadMap 是一种特定的映射方式，
    它将线程按照“warp rake”模式（即每个 warp 负责一个线性区域）分配给矩阵的行和列。
    PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL> 描述了矩阵的形状，指定了矩阵的行数和列数。
    32 表示每个 warp 有 32 个线程。
    PitchLinearShape<8, 4>：WarpThreadArrangement_,
    在 NVIDIA 的 CUTLASS 库中，WarpThreadArrangement_ 是一个模板参数，用于定义和配置一个 warp 内部线程的排列方式。
    CUTLASS 是一个高性能的 CUDA 库，专注于提供可高度配置的矩阵乘法实现，特别是在利用 Tensor Cores 的情况下。
    Thread Arrangement: 线程排列（Thread Arrangement）是指如何在一个 warp 内部排列这些线程，以便高效地执行计算任务。
    这种排列方式影响到线程如何访问数据、如何进行计算，以及如何在共享内存中组织数据。
    WarpThreadArrangement_: 这是一个模板参数，通常用于配置一个 warp 内部的线程排列。
    它通常以两个整数形式出现，例如 <M, N>，表示 warp 内线程在一个 M x N 的网格中排列。
    这种排列方式可以优化线程间的数据共享和计算效率。
    用法WarpThreadArrangement_ 的具体用法通常与 CUTLASS 的其他模板参数结合使用，以配置矩阵乘法等操作。

    优化内存访问:
    通过合理的线程排列，可以确保内存访问是共线的（coalesced），从而最大化内存带宽的利用率。
    例如，如果一个 warp 的线程排列为 <8, 4>，这意味着 warp 内的 32 个线程被排列成 8 行 4 列的形式，这样可以优化对行或列的访问。
    提高计算效率:不同的线程排列方式可以更好地适应不同大小的矩阵块，减少计算中的空闲线程。
    例如，对于特定大小的矩阵块，一个优化的线程排列可以确保每个线程都参与计算，从而提高效率。
    支持不同的矩阵布局: 根据输入矩阵的布局（如行优先或列优先），选择合适的线程排列可以显著提高性能。
    8：每个线程每次读取的 element 的数量。

    Internal details made public to facilitate introspection Iterations along each dimension (concept: PitchLinearShape)
    #include <pitch_linear_thread_map.h>
    cutlass::transform::PitchLinearWarpRakedThreadMap< Shape_, Threads, WarpThreadArrangement_, ElementsPerAccess >::Detail Struct Reference
    */
    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>,
      32, cutlass::layout::PitchLinearShape<8, 4>, 8>;
    /*
    在 GPU 编程中，理解线程如何映射到数据结构（如矩阵）上是优化性能的关键之一。
    warp rake 是一种线程映射策略，用于将 GPU 的线程 warp 分配到矩阵的行或列上，
    以便高效地利用 GPU 的计算资源和内存带宽。
    
    基本概念
    Warp:
    在 NVIDIA GPU 中，一个 warp 通常由 32 个线程组成。所有线程在一个 warp 中同步执行相同的指令。
    Rake（耙）:
    在 warp rake 策略中，warp 的线程被分配成一个“耙”形状，通常是沿着矩阵的某个维度（行或列）分布。
    这种分布方式的目标是最大化内存访问的并行性和效率。
    
    例子
    假设我们有一个矩阵 A，尺寸为 8 行 x 8 列，我们使用 warp rake 策略将一个 warp 的 32 个线程映射到这个矩阵上。
    为了简单起见，我们假设每个线程负责一个元素的计算。
    
    Warp Rake 映射
    行优先（Row-major）Warp Rake:
    假设我们将 warp 的线程分布在矩阵的行上。
    由于一个 warp 有 32 个线程，而我们的矩阵只有 8 列，所以我们可以让每个线程处理矩阵的一行。
    线程 0 到 7 负责处理矩阵的第 0 行到第 7 行。
    线程 8 到 15 处理下一部分（如果有更多行的话），依此类推。
    列优先（Column-major）Warp Rake:
    在这种映射中，warp 的线程分布在矩阵的列上。
    线程 0 到 7 负责处理矩阵的第 0 列到第 7 列。
    由于矩阵只有 8 列，剩余的线程将不会被使用，或者可以用于处理其他矩阵。
    
    优势
    内存访问效率: warp rake 通过将线程分布在矩阵的行或列上，可以确保内存访问是共线的（coalesced），这意味着内存访问是连续的，
    从而提高了内存带宽的利用率。
    计算负载均衡: 通过合理的线程分配，warp rake 可以确保计算负载在不同线程之间均匀分布，从而提高计算效率。
    */
    
    /*
    GmemIterator 是一个用于从全局内存中读取或写入数据的迭代器。
    PredicatedTileIterator 是一种迭代器类型，支持条件化的内存访问，即它可以根据需要跳过某些内存位置。
    MatrixShape<EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL> 定义了该全局内存空间矩阵的总形状。
    Element 是矩阵元素的数据类型。
    Layout 是矩阵的存储布局，比如行优先或列优先。
    1 是一个与访问顺序相关的参数。
    ThreadMap 指定了线程如何映射到矩阵块上。
    */

    using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL>,Element,Layout,1,ThreadMap>;

    /*
    params 是 GmemIterator 的参数对象，用于初始化迭代器。
    matrix.layout() 提供了矩阵的布局信息，用于配置迭代器的内存访问模式。
    */

    typename GmemIterator::Params params(matrix.layout());
    
    /*
    SmemIterator 是一个用于访问共享内存的迭代器。
    RegularTileIterator 是一种迭代器类型，用于在共享内存中以规则模式访问矩阵块。
    ColumnMajorTensorOpMultiplicandCongruous<16,64> 定义了共享内存中的数据布局，通常用于张量操作。
    其他参数与 GmemIterator 类似，指定了矩阵形状、元素类型和线程映射。
    */
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
            cutlass::MatrixShape<EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL>,Element,
            cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16,64>,1,
            ThreadMap>;
    /*
    template<int ElementSize, int Crosswise> struct cutlass::layout::TensorOpMultiplicand< ElementSize, Crosswise >
    Template based on element size (in bits) - defined in terms of pitch-linear memory and Crosswise size (in elements).

    template<int ElementSize, int Crosswise>struct cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous< ElementSize, Crosswise >
    Template mapping a column-major view of pitch-linear memory to TensorOpMultiplicand

    此处，定义16，则element 为 16位，即half_t,64 则代表横向，即一行64个元素
    */
    
    dim3 grid(1,1);
    dim3 block(32,1,1);

    int smem_size = int(sizeof(Element)*EXAMPLE_MATRIX_ROW*EXAMPLE_MATRIX_ROW);

    kernel_dump<Element,GmemIterator,SmemIterator><<<grid,block,smem_size,0>>>(params,matrix.device_ref());

    cudaError_t result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
        std::cout << "Failed" << std::endl;
    }

    return (result == cudaSuccess ? 0 : -1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
