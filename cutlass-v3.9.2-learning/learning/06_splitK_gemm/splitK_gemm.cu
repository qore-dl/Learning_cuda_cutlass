
/**
This example shows how to use split-k version of matrix multiplication using functions and data
 structures provided by CUTLASS; which we run on a NVIDIA Volta GPU.
此示例展示了如何使用 CUTLASS 提供的函数和数据结构来实现矩阵乘法的分块(split-K)版本；
我们是在 NVIDIA Volta GPU 上运行此程序的。

What is split-k?
Consider a problem size of M = 128, N = 128, K = 4096. 
In this case, if my thread-block tile size (atile can be viewed as a 2d matrix) is 128x128x4096, 
then we launch a singled a thread-block taking up a single SM of 84 SMs present on V100. 
Hence the efficiency of computation is really low. 
考虑一个M = 128, N = 128, K = 4096的问题。
在这种情况下，如果我们的thread-block对应的tile大小（tile可以看作一个2d矩阵）是128x128x4096，
那么我们启动一个单独的block，占用V100上存在的84个SMs中的某个SM。
因此，计算效率非常低。

So, how to solve it? This is where split-k comes in. 
It is a way of partitioning K-dimension of matrix multiplication 
and distribute across multiple SMs and get better efficiency than single SM.
In the above example, we can partition K-dimension with split-k factor of 16 
i.e., thread-block tile size will be 128x128x256 and will be launching on 16 SMs.
Once each thread-block computes their partial inner product (1/16th of output), 
they accumulate to single output matrix.

那么，如何解决这个问题呢？这就是“split-k”技术的用武之地。
这是一种将 K 维矩阵乘法分解的方法，
并将其分布在多个 SM 上，从而比单个 SM 更具效率。
在上述示例中，我们可以使用split-k 系数为 16 的方式对 K 维进行分解，
即block上处理的tile大小将为 128x128x256，并将在 16 个 SM 上启动。
一旦每个线程块计算出它们的部分内积（输出的 1/16），它们就会累加形成一个单一的输出矩阵。

Writing a single high performance matrix multiplication kernel is hard but do-able. 
Whereas writing high performance kernels at scale which works for multiple problem sizes with good abstractions is really hard. 
CUTLASS solves this problem by providing simplified abstractions to compose multiple sections of gemm kernel. 
When used properly, the kernels can hit peak performance of GPU easily.

编写一个高效的矩阵乘法 kernel 并非易事，但还是可以做到的。
然而，大规模编写适用于多种问题规模且具有良好抽象的高性能核心函数却非常困难。
CUTLASS 通过提供简化的抽象来组合多个 gemm kernel，从而解决了这个问题。
如果使用得当，这些核心函数能够轻松达到 GPU 的峰值性能。

CUTLASS divides a kernel into hierarchical composable sections. 
Which means, at each thread, warp and thread-block level, they compute on their own tile-size with higher level of tile sizes being composed from lower level ones.
Multiple thread-tiles (tile size each thread computes) can be used to form warp-tiles (tile size each warp computes) 
and multiple warp tiles can be used to compute threadblock-tile (tile size computed by a threadblock).

具体来说，CUTLASS 将内核划分为具有层次结构的可组合部分。
这意味着，在每个thread、warp和block级别，它们都会根据自己的tile大小进行计算，
而更高层级的tile大小则是由更低层级的tile大小组合而成的。
多个线程tile（每个线程计算的tile大小）可以组成线程束tile（每个线程束计算的tile大小），
并且多个线程束tile可以用于计算线程块tile（由一个线程块计算的tile大小）。

In this example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel can view them (logical to physical mapping)
2. Setting up computation properties: describes how the above set matrices will be used to compute output of matrix multiplication.

在这个例子中，我们将变量的初始化操作分成了几个步骤。
1. 设置数据属性：描述矩阵在内存中的布局方式以及kernel如何对其进行查看（即从逻辑到物理的映射）
2. 设置计算属性：说明上述设定的矩阵将如何用于进行矩阵乘法运算的计算过程。

First, we setup the data types of matrices A, B, C and D along with alpha, beta 
as the equation for GEMM is D = alpha * A * B + beta * C. 
In CUTLASS, the kernels first compute A * B and leaves the rest of the computation to end of the kernel 
as alpha * X + beta * C is a simple element-wise operation on X (A * B) and C. 
We call this as epilogue of kernel.
Hence, we setup data types for alpha and beta to be equal to ElementComputeEpilogue = float. 
As we want to MMA instructions on Volta and they support only half-precision floating point (fp16 or half), 
we use data type for elements in input matrix A and B as cutlass::half_t. 
Volta also supports accumulation of partial dot product to fp32, which can store wider range of numbers, we use it as data type of output matrix elements and accumulation.

首先，我们设定矩阵 A、B、C 和 D 以及参数 alpha 和 beta 的数据类型；
GEMM 方程为 D = alpha * A * B + beta * C。
在 CUTLASS 中，内核首先计算 A * B，其余的计算则留到内核的末尾进行，
因为 alpha * X + beta * C 是对 X（A * B）和 C 进行的简单元素级运算（标量乘法和元素级别加法）。
我们称之为内核的末尾（epilogue）。
因此，我们为 alpha 和 beta 设定了相同的数据类型，即 ElementComputeEpilogue = float。
由于我们希望在 Volta 上使用矩阵乘法指令，并且它们仅支持半精度浮点数（fp16 或半精度），
我们为输入矩阵 A 和 B 中的元素使用 cutlass:：half_t 这种数据类型。
Volta 还支持将部分点积累加到 fp32 中，这可以存储更广泛的数值范围，因此我们将它用作输出矩阵元素和累加的数据类型。

We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). 
Communicating just the data type is not enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. 
We do that by initializing template variable LayoutInputA to column major cutlass variable, 
LayoutInputB to row major and LayoutOutput to row major. 
Next, we setup rules to compute alpha * X + beta * C which is called epilogue of the kernel.
We initialize template variable EpilogueOp, which takes the data type of output ElementOutput (float), 
the number of elements per vector memory access (16), data type of accumulator (float) 
and data type of computation of linear combination (alpha * X + beta * C).
我们通过初始化模板变量来向 CUTLASS 内核传递这些信息：
ElementAccumulator（float）、ElementComputeEpilogue（float）、ElementInputA（half_t）、
ElementInputB（half_t）、ElementOutput（float）。
仅仅传递数据类型是不够的。由于数据在内存中是以线性方式排列的，所以我们必须传递矩阵的布局信息。
我们通过将模板变量 LayoutInputA 初始化为列主序的cutlass 变量、LayoutInputB 初始化为行主序以及 LayoutOutput 初始化为行主序来实现这一点。
接下来，我们设置计算规则以计算 alpha * X + beta * C，这被称为内核的后处理部分。
我们初始化模板变量 EpilogueOp，它接受输出数据类型 ElementOutput（float）、每个向量内存访问的数据元素数量（16）、累加器的数据类型（float）
以及线性组合的数据类型（alpha * X + beta * C）。

Now that we setup the properties of data, we have to setup properties of computation.
现在我们设置了数据的属性，我们必须设置计算的属性。

Second, we create template variables of tile sizes for thread-block, warp and (thread)mma-op to 128x128x32, 64x64x4, 8x8x4 (MxNxK) respectively. 
When passed to instantiate CUTLASS GEMM kernel, 
it internally deduce the amount of threads needed per thread-block, amount of shared memory, storing data in bank-conflict free manner, 
and ton of other variables required to compose, initialize and launch a high performance GEMM kernel. 
This is the beauty of CUTLASS, it relieves developer from understanding and coding complicated hardware optimizations which can easily go wrong.
其次，我们为block、warp和（线程）矩阵乘法操作创建了相应的tile大小模板变量，分别为 128x128x32、64x64x4、8x8x4（MxNxK）。
当传递这些模版变量来初始化 CUTLASS 矩阵乘法 kernel 函数时，
它会自动推算出每个block所需的线程数量、shared memory的大小、以避免shared memory bank 冲突的方式存储数据，
以及构建、初始化和启动高性能矩阵乘法核函数所需的大量其他变量。
这就是 CUTLASS 的魅力所在，它让开发者无需理解并编写复杂的硬件优化代码，这些代码很容易出错。

There are few more template variables initialized such as, which threadblock tile of output matrix is done
 which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.
还有一些其他的模板变量是需要进行初始化的，比如输出矩阵的哪个线程块区域已完成、
哪个线程块在哪个 SM 上启动、您想要运行的 GPU 的 CUDA SM 架构类型等。

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::GemmSplitKParallel template.
所有这些内容都被整合在一起，形成了一个模板变量，
该变量使用“cutlass::gemm::device::GemmSplitKParallel”模板来描述“CUTLASS 矩阵乘法核”。

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.
下一步是初始化物理数据、实例化和初始化CUTLASS内核并运行它。
我们使用CUTLASS实用程序来初始化，填充，比较矩阵，因为它们很简单，并且不妨碍学习CUTLASS。

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the important one, split k-dimension factor. 
Along with that, we query CUTLASS if any scratch-space memory required by the kernel we instantiated. 
If yes, we create it and pass it along with other arguments created to initialize CUTLASS kernel then, 
the kernel is launched.

一旦所有矩阵都初始化并填充了数据，创建参数元组以启动CUTLASS内核，该内核接受问题大小（M = 5120, N = 4096和K = 4096），矩阵，alpha， beta和重要的一个，分割K维因子。
除此之外，如果我们实例化的内核需要任何抓取空间内存，我们将查询CUTLASS。
如果是，我们创建它，并将它与为初始化CUTLASS内核而创建的其他参数一起传递，然后启动内核。

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if the output from CUTLASS kernel is same as reference GEMM kernel.
在这个示例中，我们稍后启动一个参考gemm内核（来自CUTLASS utilities 程序），以比较来自CUTLASS内核的输出是否与参考gemm内核相同。
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
// #include "../common/helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Put all the created template variables to create GemmSplitKParallel template variable
using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    ElementInputA,
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
    EpilogueOp>;

int run(){
    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props,0);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (props.major < 7) {
        std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
                  << std::endl;
    
        // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
        return 0;
    }

    //
    // Define problem size
    //

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
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn());  // <- Create matrix C with dimensions M x N
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
        tensor_c.host_view(),
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
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 16 partitions
    int split_k_slices = 16;
    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
        tensor_a.device_ref(),  // <- reference to matrix A on device
        tensor_b.device_ref(),  // <- reference to matrix B on device
        tensor_c.device_ref(),  // <- reference to matrix C on device
        tensor_d.device_ref(),  // <- reference to matrix D on device
        {alpha, beta},          // <- tuple of alpha and beta
        split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Initialize CUTLASS kernel with arguments and workspace pointer
    cutlass::Status status = gemm_op.initialize(arguments,workspace.get());
    // CUTLASS_CHECK(status);
    
    // Launch initialized CUTLASS kernel
    status = gemm_op();
    // CUTLASS_CHECK(status);

    // Create instantiation for device reference gemm kernel
    cutlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue> gemm_device;
    
    gemm_device(problem_size,
                alpha,
                tensor_a.device_ref(),
                tensor_b.device_ref(),
                beta,
                tensor_c.device_ref(),
                tensor_ref_d.device_ref());
    
    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        tensor_ref_d.host_view());
    
    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0  : -1);
    
}

int main() {

    //
    // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
    //
    // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
    //
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
      std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;
  
      // Returning zero, so this test passes when built with older CUDA Toolkits. Its action are no-op.
      return 0;
    }
    else {
      return run();
    }
}






