#include <iostream>
#include <sstream>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
/*
定义输入、输出矩阵以及输入矩阵元素间运算的数据类型
*/

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// Note that if the output is column major, the bias has to be per row. i.e. every row has different bias.
// If the output is row major, the bias has to be per column, i.e. every column has different bias.
// Below list some other notices:
//
// Note this example only works for ColumnMajor output because
//   1) we only have row major epilogue.
//   2) we swap A and B if the output is column major then we can still use the row major epilogue.
//   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
//   4) we can use the existing OutputIterator to load 1xM bias vector.

//注意，如果输出是列为主，则偏置必须为每行。也就是说，每一行都有不同的偏差。
//如果输出为行为主，则偏置必须为每列，即每列有不同的偏置。
//下面列出了一些其他注意事项：
//
//注意这个例子只适用于ColumnMajor输出，因为
// 1)我们只有行主序的尾部计算。
// 2)我们交换A和B，如果输出是列为主的，那么我们仍然可以使用行为主的尾部计算。
// 3) Mx1偏置向量在交换/转置后变为1xM。
// 4)我们可以使用现有的OutputIterator来加载1xM的偏置向量。

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
/*
 这段代码描述了每个block将计算的tile大小
*/
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128,128,32>; // <- threadblock tile M = 128, N = 128, K = 32

// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
/*
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle 是 NVIDIA CUTLASS 库中的一个类，用于定义线程块在执行矩阵乘法（GEMM）时的调度策略。CUTLASS 是一个专为 NVIDIA GPU 提供高性能矩阵计算的库，支持灵活的配置和优化。


什么是 Threadblock Swizzle？
在 CUDA 编程中，线程块（thread block）是执行并行计算的基本单元。
在矩阵乘法中，线程块的调度策略（swizzle）决定了如何将计算任务分配到不同的线程块上。
不同的调度策略可以影响计算的性能和资源利用率。


GemmIdentityThreadblockSwizzle 的作用
GemmIdentityThreadblockSwizzle 是一种简单的调度策略，通常用于对线程块进行直接和线性的映射。其主要特征包括：


直接映射：
该策略直接将线程块映射到计算网格（grid）中，不进行复杂的重新排列。
这意味着每个线程块按照其在线性网格中的位置来处理对应的计算任务。
适用场景：
适用于计算密集型任务，尤其是当矩阵大小和线程块配置已经被优化以充分利用 GPU 资源时。
在这种情况下，简单的映射策略可以减少调度开销，并提高计算效率。
实现简单：
由于其直接映射的特性，GemmIdentityThreadblockSwizzle 的实现相对简单，不需要复杂的计算来确定每个线程块的任务。

具体实现
在 CUTLASS 中，GemmIdentityThreadblockSwizzle 通常是通过模板参数来配置的，
允许用户指定线程块的大小和其他相关参数。其基本工作流程包括：


初始化：
配置线程块大小和网格大小。
设置矩阵的维度和数据布局。
任务分配：
将每个线程块直接分配给计算网格中的一个位置。
每个线程块负责处理矩阵乘法中的一个子矩阵。
并行计算：
启动 CUDA 核函数，在线性网格中执行线程块。
每个线程块独立计算其负责的子矩阵乘法。

优势和局限
优势：
实现简单，调度开销低。
在矩阵大小和线程块配置合理时，能够提供良好的性能。
局限：
对于不规则的矩阵大小或需要复杂调度的场景，可能不是最优选择。
可能无法充分利用所有的 GPU 资源，尤其是在矩阵大小与线程块配置不匹配时。

总之，cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle 是一种简单且直接的线程块调度策略，
适用于需要高效执行的标准矩阵乘法任务。在使用 CUTLASS 进行矩阵乘法优化时，
选择合适的线程块调度策略可以显著影响计算性能。
*/
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
/*
简而言之，上面定义了如何把grid中的block 映射或者放置在grid中处理不同的数据。
此时，线程块是按照编号线性放置在计算网格中的。这意味着每个线程块的编号直接对应于其在网格中的位置，没有复杂的重新排列或映射。


线性放置的含义
直接映射：线程块的编号与其在网格中的位置直接对应。例如，编号为 0 的线程块放置在网格的第一个位置，编号为 1 的线程块放置在第二个位置，依此类推。
简单调度：这种线性放置策略非常简单，因为它不需要额外的计算来确定线程块的位置。这减少了调度的复杂性和开销。
适用场景：这种策略适合于矩阵大小和线程块配置已经优化的情况，因为简单的线性映射可以有效利用 GPU 的计算资源。

例子
假设我们有一个 2D 网格用于执行矩阵乘法，网格的尺寸为 
(grid_dim.x,grid_dim.y)，那么在线性放置策略下：

线程块 (0, 0) 对应网格中的第一个位置。
线程块 (1, 0) 对应网格中的第二个位置。
线程块 (0, 1) 对应网格中的第三个位置。
依此类推。

这种线性放置策略使得线程块的调度和管理变得简单高效，尤其是在不需要复杂调度的情况下。
它能够为大多数标准矩阵乘法任务提供良好的性能表现。
*/

/*
接下来定义kernel尾部操作：LinearCombinationRelu;
*/
// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

// Number of pipelines you want to use
constexpr int NumStages = 2;
/*
constexpr 是 C++11 引入的一个关键字，用于在编译时计算常量表达式。它不属于 C 语言的标准，因此在纯 C 语言中是不存在的。不过，我可以解释一下 constexpr 在 C++ 中的作用，以帮助你理解它在编译时计算中的角色。


constexpr 在 C++ 中的作用
编译时常量：constexpr 关键字用于声明可以在编译时求值的常量。这意味着编译器会在编译阶段计算出结果，而不是在运行时。这对于性能优化和代码安全性有帮助，因为编译时常量可以用于数组大小、模板参数等需要常量表达式的地方。
函数和变量：constexpr 可以用于修饰变量和函数。
变量：constexpr 变量必须在定义时初始化，并且初始化表达式必须是编译时可计算的。
函数：constexpr 函数可以在编译时求值，如果其参数也是常量表达式的话。这样的函数必须有一个返回值，并且函数体中只能包含单一的可执行语句（在 C++11 中，C++14 开始放宽了这一限制）。
示例
constexpr int square(int x) {
    return x * x;
}

constexpr int size = square(4);  // 在编译时计算出 size 的值为 16

int array[size];  // 使用编译时常量作为数组大小
*/

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

int run(){
    const int length_m = 5120;
    const int length_n = 4096;
    const int length_k = 4096;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m,length_n,length_k);
    
    // Initialize tensors using CUTLASS helper functions
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk());  // <- Create matrix A with dimensions M x K
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn());  // <- Create matrix B with dimensions K x N

    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias(
        {problem_size.m(), 1});  // <- Create matrix C with dimensions M x 1
    
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
            problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                 // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
            problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                                 // reference kernel
      
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        ElementInputA(4),
        ElementInputA(-4),
        0);  // <- Fill matrix A on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0);  // <- Fill matrix B on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c_bias.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0);  // <- Fill matrix C on host with uniform-distribution random data
    cutlass::reference::host::TensorFill(
        tensor_d.host_view());  // <- fill matrix D on host with zeros
    cutlass::reference::host::TensorFill(
        tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros
      
    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c_bias.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size,                       // <- problem size of matrix multiplication
        tensor_a.device_ref(),              // <- reference to matrix A on device
        tensor_b.device_ref(),              // <- reference to matrix B on device
    
        {tensor_c_bias.device_data(), 0},   // <- the C matrix is treated as the bias vector. 
        // We can enable the GEMM to project away the N dimension by setting the stride to zero.
    
        tensor_d.device_ref(),              // <- reference to matrix D on device
        {alpha},                              // <- alpha
        split_k_slices};                    // <- k-dimension split factor
    
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments,workspace.get());

    // Launch initialized CUTLASS kernel
    status = gemm_op();

    //
    // Create instantiation for device reference gemm kernel
    //

    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue> gemm_device_reference;
    
    gemm_device_reference(
        problem_size,
        alpha,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        0,
        tensor_ref_d.device_ref());
    
    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Compute bias + relu in host code
    for(int i=0;i<problem_size.m();++i){
        for(int j=0;j<problem_size.n();++j){
            tensor_ref_d.at({i,j}) = std::max(ElementOutput(0),
            ElementOutput(tensor_ref_d.at({i,j})+tensor_c_bias.at({i,0})));
        }
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    std::cout<<(cutlass::reference::host::TensorEquals(tensor_d.host_view(),tensor_ref_d.host_view())? "Passed" : "Failed")<<std::endl;

    return 0;
}

int main(){
    bool notSupported = false;

    // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
    //
    // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props,0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if(!((props.major*10 + props.minor)>=75)){
        std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
              << std::endl;
        notSupported = true;
    }

    if(notSupported){
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    return run();
}




