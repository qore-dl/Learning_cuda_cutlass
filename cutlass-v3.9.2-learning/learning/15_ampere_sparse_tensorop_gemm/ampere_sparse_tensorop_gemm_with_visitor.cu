/**
Please check example 07, 08 and 17 for the basics of dense tensor op gemm kernels.  NVIDIA Ampere
architecture also supports structured sparse tensor op for tf32, fp16, int8 and int4.
Sparse GEMM kernels needs to takes an additional E matrix which stores the meta data.  The format of
meta data is different for every data types.   CUTLASS templates can automatically infer it based on
input A and B.  Check code below.
Moreover, matrix E needs to be preprocessed so that it can use ldmatrix to load into the registers
efficiently.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse_with_visitor.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = int8_t;                       // <- data type of elements in input matrix A
using ElementInputB = int8_t;                       // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// The number of elements per vectorized memory access. 
constexpr int AlignmentInputA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
constexpr int AlignmentInputB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
constexpr int AlignmentComputeEpilogue = 128 / cutlass::sizeof_bits<ElementComputeEpilogue>::value;
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 128>;  // <- threadblock tile M = 128, N = 128, K = 128
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 128
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;  // <- MMA Op tile M = 16, N = 8, K = 64

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using Operator = cutlass::arch::OpMultiplyAddSaturate;

// Number of pipelines you want to use
constexpr int NumStages = 3;

constexpr auto NumEVTEpilogueStages = 1;

/*
在 CUTLASS 库中，VisitorAuxLoad 和 VisitorAuxStore 是用于处理后处理阶段（epilogue）的辅助加载和存储操作的类。
它们的主要作用是在矩阵乘法或卷积运算的最终阶段，处理与主计算数据（如累积结果）相关的辅助数据（Auxiliary Data），
如偏置（bias）、激活函数参数等。
*/

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

using BiasTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ElementComputeEpilogue,
    AlignmentComputeEpilogue,
    NumEVTEpilogueStages>;

/*
VisitorAuxLoad 用于从内存中加载辅助数据。这些数据通常不是主计算的直接结果，但在后处理阶段需要用到，例如偏置或其他需要加到计算结果上的修正数据。
它定义了如何根据线程映射加载这些辅助数据，以便后续的计算可以使用。
典型用途:
加载偏置：在计算完成后，将偏置数据加载到寄存器中，以便与计算结果进行合并。
加载其他参数：例如用于激活函数的参数，或其他需要在后处理阶段应用的修正参数。
*/

using Bias = cutlass::epilogue::threadblock::VisitorAuxLoad<
    BiasTileThreadMap,
    ElementComputeEpilogue,
    cute::Stride<int64_t, cute::_1, int64_t>>;

/*
VisitorAuxLoad 是一个辅助加载器，用于在线程块级别加载辅助数据（如偏置）。
BiasTileThreadMap 定义了线程如何在不同的偏置数据块上进行映射。
ElementComputeEpilogue 是计算后处理时使用的数据类型。
cute::Stride<int64_t, cute::_1, int64_t> 定义了加载偏置时的步幅（即如何从内存中读取数据的模式）
*/

using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, ElementComputeEpilogue, ElementComputeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

/*
这部分定义了一个类型 ApplyBias，它是 VisitorCompute 的实例。
VisitorCompute 用于在加载的结果上执行具体的计算操作。
cutlass::plus 指定了计算操作为加法，用于将偏置加到累积结果上。
ElementComputeEpilogue 是输入和输出的数据类型。
cutlass::FloatRoundStyle::round_to_nearest 指定了舍入方式为四舍五入到最近的整数
*/

using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
    ApplyBias,
    Accum,
    Bias>;

/*
这部分定义了一个类型 EVTApplyBias，它是 Sm80EVT 的实例。
Sm80EVT 是针对 NVIDIA Ampere 架构（SM80）优化的后处理操作。
ApplyBias 是之前定义的用于执行加法的计算类型。
Accum 是累积结果的数据类型。
Bias 是之前定义的用于加载偏置的类型
*/

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ElementOutput,
    AlignmentOutput,
    NumEVTEpilogueStages>;

/*
VisitorAuxStore 用于将处理后的结果（包括主计算结果和应用了辅助数据的结果）存储回显存。
它确保计算后的数据按照正确的格式和位置存储，以便后续的计算或输出使用。
典型用途:
存储加了偏置的结果：计算结果与偏置合并后，将最终结果存储到输出矩阵中。
存储经过激活函数处理后的结果：在应用激活函数后，将结果存储到合适的位置。
*/

/*
什么是 Aux？
Aux 是 Auxiliary 的缩写，意为“辅助的”或“补充的”。
在计算中，Aux 数据通常指那些不是直接从主要计算得出的数据，但在后处理阶段需要用到的数据。
这些数据可能包括偏置、激活函数参数、缩放因子等，通常用于调整或修正计算结果。

通过使用 VisitorAuxLoad 和 VisitorAuxStore，CUTLASS 提供了一种灵活且高效的方式来处理这些辅助数据，
确保后处理阶段的计算能够高效地与主计算结果结合。


*/

using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, int64_t>>;

using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
    Output,
    EVTApplyBias>;

/*
cutlass::epilogue::threadblock::Sm80EVT 是 CUTLASS 库中的一个模板类，
用于在 NVIDIA Ampere 架构（SM80）上优化执行 epilogue 阶段的操作。
Epilogue 阶段通常是指在矩阵乘法或卷积运算之后进行的后处理操作，比如加上偏置、应用激活函数等。
*/

/*
什么是 EVT？
EVT 是 CUDA Epilogue Visitor Template 的缩写。
它是一种设计模式或模板，用于在 epilogue 阶段灵活地应用各种操作。
EVT 的设计允许我们定义一系列“访问者”（visitors），
这些访问者可以在 epilogue 阶段访问和修改计算的结果。


Sm80EVT 的作用
优化目标: Sm80EVT 专门针对 NVIDIA Ampere 架构（SM80）进行了优化。
Ampere 架构引入了一些新的硬件特性，比如更强大的张量核心（Tensor Cores），
这些特性可以被利用来提高 epilogue 阶段的性能。

灵活性: 通过使用 EVT 模式，Sm80EVT 提供了一种灵活的方式来组合和应用多种后处理操作。
你可以定义多个“访问者”，每个访问者执行特定的操作，比如加上偏置、应用激活函数、执行舍入操作等。

高效性: Sm80EVT 通过利用 Ampere 架构的硬件特性，能够在保持灵活性的同时，提供高效的计算性能。
这对于需要在大规模数据上执行复杂后处理操作的应用程序（如深度学习模型的推理和训练）尤为重要。

使用场景
在实际使用中，Sm80EVT 可能会结合多个类型的访问者来实现复杂的 epilogue 操作。
例如，在一个深度学习模型的推理过程中，你可能需要：

将矩阵乘法的结果加上一个偏置。
应用一个激活函数，比如 ReLU。
对结果进行量化或舍入。

通过 Sm80EVT，可以高效地将这些操作组合在一起，并在 Ampere 架构上以最佳性能执行。
*/

// Use element type in EVT with the smallest bitwidth as ElementC.
using ElementC = ElementComputeEpilogue;
using LayoutC = LayoutOutput;

using Gemm =
    typename cutlass::gemm::device::SparseGemmWithVisitor<
      ElementInputA, LayoutInputA,
      ElementInputB, LayoutInputB,
      ElementC, LayoutC,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EVTOutput,
      SwizzleThreadBlock,
      NumStages,
      AlignmentInputA,
      AlignmentInputB,
      Operator,
      NumEVTEpilogueStages>;

// Data type and layout of meta data matrix E can be inferred from template Gemm.
using ElementInputE = typename Gemm::GemmKernel::ElementE;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = typename Gemm::GemmKernel::LayoutE;

// Blow property is defined in include/cutlass/arch/sp_mma_sm80.h
// 50% Sparsity on Ampere
constexpr int kSparse = Gemm::kSparse;
// How many elements of A are covered per ElementE
constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
// The size of individual meta data 
constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;

int run(){

    const int length_m = 512;
    const int length_n = 512;
    const int length_k = 1024;
    
    // Create a tuple of problem size for matrix multiplication

    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    
    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse));  // <- Create matrix A with dimensions M x (K / 2)
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a_uncompressed(
        problem_size.mk());  // <- Create uncompressed matrix A with dimensions M x K for reference computing
    
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn());  // <- Create matrix B with dimensions K x N
    cutlass::HostTensor<ElementComputeEpilogue, LayoutOutput> tensor_c(
        problem_size.mn());  // <- Create matrix C with dimensions M x N
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                 // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                 // reference kernel

    // Create matrix E with dimensions M x (K / 2 / kElementsPerElementE). This one is used by reference computing.
    cutlass::HostTensor<ElementInputE, LayoutInputE> tensor_e(
        cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    // Same size as the above.  The above one needs to be reordered and stored in this one.
    cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
        cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(8),
        ElementInputA(-8),
        0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(8),
        ElementInputB(-8),
        0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(8),
        ElementOutput(-8),
        0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomSparseMeta(                                           
        tensor_e.host_view(),
        1,
        kMetaSizeInBits);   // <- Fill matrix E on host with uniform-distribution random meta data
    cutlass::reference::host::TensorFill(
            tensor_d.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
            tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros
      
    // Reorder the meta data matrix so that we can use ldmatrix to load them to tensor core
    // instructions.
    cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e.host_ref(),                         
                        {problem_size.m(), problem_size.n(),                                        
                         problem_size.k() / kSparse / kElementsPerElementE});

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_e_reordered.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);

    typename Bias::Arguments bias_arguments{
        tensor_c.device_data(),
        ElementComputeEpilogue(0),
        {problem_size.n(), cute::_1{}, problem_size.mn().product()}
      };
    typename Output::Arguments output_arguments{
        tensor_d.device_data(),
        {problem_size.n(), cute::_1{}, problem_size.mn().product()}
    };
    typename EVTOutput::Arguments callback_arguments{
        {
          {},                    // Accum
          bias_arguments,        // Bias
          {}                     // ApplyBias
        },                       // EVTApplyBias
        output_arguments         // Output
    };                         // EVTOutput

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
        tensor_a.device_ref(),  // <- reference to matrix A on device
        tensor_b.device_ref(),  // <- reference to matrix B on device
        tensor_e_reordered.device_ref(),  // <- reference to matrix E on device
        callback_arguments};    // <- epilogue arguments

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());

    // Launch initialized CUTLASS kernel
    status = gemm_op();

    // uncompress tensor_a based on meta data tensor_e. We need it for reference computing.
    cutlass::uncompress(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
                      tensor_e.host_ref(), problem_size.m(), problem_size.k());

    // Create instantiation for host reference gemm kernel
    cutlass::reference::host::Gemm<ElementInputA,
                                 LayoutInputA,
                                 ElementInputB,
                                 LayoutInputB,
                                 ElementOutput,
                                 LayoutOutput,
                                 ElementComputeEpilogue,
                                 ElementComputeEpilogue,
                                 typename Gemm::Operator> gemm_host;
    
    // Launch host reference gemm kernel
    gemm_host(problem_size,
        alpha,
        tensor_a_uncompressed.host_ref(),
        tensor_b.host_ref(),
        beta,
        tensor_c.host_ref(),
        tensor_ref_d.host_ref());
    
    // Copy output data from CUTLASS host for comparison
    tensor_d.sync_host();
    
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_ref_d.host_view());
    
    std::cout << (passed ? "Passed" : "Failed") << std::endl;
    
    return (passed ? 0  : -1);
}

int main(){
    
    bool notSupported = false;
    // Ampere Sparse Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.1. 
    //
    // CUTLASS must be compiled with CUDA 11.1 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1))) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.1 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (props.major * 10 + props.minor < 80) {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
        notSupported = true;
    }

    if (notSupported) {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    return run();
}



















































































































































































