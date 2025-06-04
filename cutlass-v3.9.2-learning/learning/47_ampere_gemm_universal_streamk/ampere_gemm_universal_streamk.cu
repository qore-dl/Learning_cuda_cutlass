/***************************************************************************************************
 Example contrasting the Stream-K parallel decomposition for GEMM threadblocks versus the
 "classic data-parallel" and "Split-K" decompositions.

 本例子旨在将GEMM线程块的 Stream-K 并行分解与 “经典数据并行”和 “Split-K” 分解进行对比。

 For more details regarding the Stream-K method, see "Stream-K: Work-centric Parallel Decomposition
 for Dense Matrix-Matrix Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)

 有关Stream-K方法的更多细节，请参见："Stream-K: Work-centric Parallel Decomposition
 for Dense Matrix-Matrix Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)

 Requires NVIDIA Ampere or newer device (SM80+).

 - To lock persistence mode, power (400W), clocks (1005MHz) for evaluation (assumes device 0 and A100)

 需要NVIDIA安培或更新的设备（SM80+）。
 - 锁定持久模式，电源（400W），时钟（1005MHz）用于评估（假设使用 GPU 0，型号为A100）

     cutlass$ sudo nvidia-smi -pm 1 -i 0

     cutlass$ sudo nvidia-smi -i 0 -pl 400

     cutlass$ sudo nvidia-smi -i 0 -lgc 1005

 - 编译和运行方法如下：
 - Build and run:

     cutlass$ mkdir build

     cutlass$ cd build

     cutlass/build$ cmake .. -DCUTLASS_NVCC_ARCHS=80

     cutlass/build$ make 47_ampere_gemm_universal_streamk

     cutlass/build$ ./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk

        10000 timing iterations of 2048 x 2048 x 2048 matrix-matrix multiply

        Basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.112633 ms
          GFLOPs: 152530

        StreamK GEMM with default load-balancing
          Disposition: Passed
          Avg runtime: 0.0941929 ms
          GFLOPs: 182390
          Speedup vs Basic-DP: 1.196

        StreamK emulating basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.113119 ms
          GFLOPs: 151875
          Speedup vs Basic-DP: 0.996

        Basic split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.104772 ms
          GFLOPs: 163973

        StreamK emulating Split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.105379 ms
          GFLOPs: 163029
          Speedup vs Basic-SplitK: 0.994

 **************************************************************************************************/

#include <iostream>
#include <string>
 
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
 
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "../common/helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA = cutlass::half_t;                                // Element type for A matrix operand
using LayoutA  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
// Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;    

// B matrix configuration
using ElementB = cutlass::half_t; // Element type for B matrix operand
using LayoutB = cutlass::layout::RowMajor; // Layout type for B matrix operand
// Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// C/D matrix configuration
using ElementC = cutlass::half_t; // Element type for C/D matrix operand
using LayoutC = cutlass::layout::RowMajor; // Layout type for C/D matrix operand
// Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator = cutlass::half_t;                          // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;           // Operator class tag
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;     // Warp-level tile size (concept: GemmShape)
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;      // Instruction-level tile size (concept: GemmShape)
// Number of global->shared pipeline stages used in the GEMM mainloop
/*
Number of Global->Shared Pipeline Stages
在 GEMM 的实现中，数据通常从全局内存（global memory）加载到共享内存（shared memory），
然后再从共享内存加载到寄存器中进行计算。这个过程之所以需要优化，是因为全局内存访问的延迟较高，
而共享内存的访问速度要快得多。


Number of Global->Shared Pipeline Stages 指的是在 GEMM 主循环（main loop）中，
从全局内存到共享内存的数据传输过程中使用的流水线阶段（pipeline stages）的数量。
这些阶段的目的主要是为了隐藏全局内存访问的延迟，从而提高整体计算效率。
*/
constexpr int NumStages = 4;  

// Epilogue output operator
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, // Element type for C and D matrix operands
    AlignmentC, // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator, // Element type from internal accumaccumulation
    ElementAccumulator>; // Data type used to compute linear combination (alpha,beta)

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    ElementAccumulator>;

/*
定义了不同的Threadblock(CTA) 与 Grid/SM Mapping 的策略下的GemmUniversal 模版。
以便在实验中定义不同类型的kernel。
GemmUniversal 是一个有状态的、可重用的 GEMM 句柄。一旦为给定的 GEMM 计算（问题几何形状和数据引用）初始化后，
它就可以在具有相同几何形状的不同 GEMM 问题之间重复使用。
（一旦初始化，有关问题几何形状和指向工作区内存的引用的详细信息将无法更新。）
通用 GEMM 支持串行归约、并行归约、批量跨步和批量数组变体。
GemmUniversal的具体代码可见：include/cutlass/gemm/device/gemm_universal.h
    其继承自模版类 GemmUniversalBase（include/cutlass/gemm/device/gemm_universal_base.h），
    因此主要实现都在 GemmUniversalBase 中。
    GemmUniversalBase需要传入一个kernel级的具体GEMM实现类：
    template <typename GemmKernel_>
    class GemmUniversalBase

    GemmUniversalBase的默认模版参数GemmKernel(GemmUniversalBase::GemmKernel) 为 cutlass::gemm::kernel::DefaultGemmUniversal::GemmKernel。
    后者根据传入的模板参数ThreadblockSwizzle来确定。
    

    GemmUniversal 整体定义如下：
    template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for.  This is the minimum SM that
    /// supports the intended feature. The device kernel can be built
    /// targeting any SM larger than this number.
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB = ComplexTransform::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout_ = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout_ = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout_ = layout::NoPermute
    >
    class GemmUniversal : 
    public GemmUniversalBase<
        typename kernel::DefaultGemmUniversal<
        ElementA_,
        LayoutA_,
        TransformA,
        AlignmentA,
        ElementB_,
        LayoutB_,
        TransformB,
        AlignmentB,
        ElementC_,
        LayoutC_,
        ElementAccumulator_,
        OperatorClass_,
        ArchTag_,
        ThreadblockShape_,
        WarpShape_,
        InstructionShape_,
        EpilogueOutputOp_,
        ThreadblockSwizzle_,
        Stages,
        Operator_,
        SharedMemoryClearOption::kNone,
        GatherA,
        GatherB,
        ScatterD,
        PermuteDLayout_,
        PermuteALayout_,
        PermuteBLayout_
        >::GemmKernel
    > {

    public:

    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    using PermuteDLayout = PermuteDLayout_;
    using PermuteALayout = PermuteALayout_;
    using PermuteBLayout = PermuteBLayout_;
    static int const kStages = Stages;
    static int const kAlignmentA = AlignmentA;
    static int const kAlignmentB = AlignmentB;
    static int const kAlignmentC = EpilogueOutputOp::kCount;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    
    
    
    using Base = GemmUniversalBase<
        typename kernel::DefaultGemmUniversal<
        ElementA_,
        LayoutA_,
        TransformA,
        AlignmentA,
        ElementB_,
        LayoutB_,
        TransformB,
        AlignmentB,
        ElementC_,
        LayoutC_,
        ElementAccumulator_,
        OperatorClass_,
        ArchTag_,
        ThreadblockShape_,
        WarpShape_,
        InstructionShape_,
        EpilogueOutputOp_,
        ThreadblockSwizzle_,
        Stages,
        Operator_,
        SharedMemoryClearOption::kNone,
        GatherA,
        GatherB,
        ScatterD,
        PermuteDLayout_,
        PermuteALayout_,
        PermuteBLayout_
        >::GemmKernel
    >;

    using Arguments = typename Base::Arguments;
    using GemmKernel = typename Base::GemmKernel;
};

*/

// Classic data-parallel device GEMM implementation type
using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
    ElementA,LayoutA,
    ElementB,LayoutB,
    ElementC,LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

// StreamK device GEMM implementation type
using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversal<
    ElementA,LayoutA,
    ElementB,LayoutB,
    ElementC,LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result{
    double avg_runtime_ms;
    double gflops;
    cutlass::Status status;
    cudaError_t error;
    bool passed;

    Result(
        double avg_runtime_ms = 0,
        double gflops = 0,
        cutlass::Status status = cutlass::Status::kSuccess,
        cudaError_t error = cudaSuccess):
            avg_runtime_ms(avg_runtime_ms),gflops(gflops),status(status),error(error),passed(true) 
        {}
};

/// Command line options parsing
/*
创建一个 Options 结构体。
*/
struct Options{
    std::string command_name;
    bool help;
    cutlass::gemm::GemmCoord problem_size;
    float alpha;
    float beta;
    int split_k_factor;
    int avail_sms;
    bool reference_check;
    int iterations;

    cutlass::HostTensor<ElementA,LayoutA> tensor_a;
    cutlass::HostTensor<ElementB,LayoutB> tensor_b;
    cutlass::HostTensor<ElementC,LayoutC> tensor_c;
    cutlass::HostTensor<ElementC,LayoutC> tensor_d;
    cutlass::HostTensor<ElementC,LayoutC> tensor_ref_d;

    Options(std::string command_name):
        command_name(command_name),
        help(false),
        problem_size({2048,2048,2048}),
        alpha(1.0),
        beta(0.0),
        split_k_factor(1),
        avail_sms(-1),  // Number of device SMs to use is unlimited
        reference_check(true),
        iterations(10000)
    {}

    bool valid const
    {
        return true;
    };
    /*
    Options::parse 通过 CommandLine 结构体解析命令行参数。
    */
    void parse(int argc,char const **args){
        cutlass::CommandLine cmd(argc,args);

        if(cmd.check_cmd_line_flag("help")) {
            help = true;
        }

        cmd.get_cmd_line_argument("m",problem_size.m());
        cmd.get_cmd_line_argument("n",problem_size.n());
        cmd.get_cmd_line_argument("k",problem_size.k());
        cmd.get_cmd_line_argument("alpha",alpha);
        cmd.get_cmd_line_argument("beta",beta);
        cmd.get_cmd_line_argument("split",split_k_factor);
        cmd.get_cmd_line_argument("iterations",iterations);
    }

    std::ostream & print_usage(std::ostream &out) const
    {
        out
            << "Performs a GEMM computation.\n"
            << "\n"
            << "Options:\n"
            << "\n"
            << "  --help                      If specified, displays this usage statement.\n\n"
            << "  --m=<int>                   GEMM M dimension\n"
            << "  --n=<int>                   GEMM N dimension\n"
            << "  --k=<int>                   GEMM K dimension\n"
            << "  --alpha=<f32>               Epilogue scalar alpha\n"
            << "  --beta=<f32>                Epilogue scalar beta\n\n"
            << "  --split=<int>               Split-K factor to emulate\n\n"
            << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

        out
            << "\n\nExamples:\n\n"
            << "$ " << command_name << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";
    
        return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const{
    // Two flops per multiply-add
    return (2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Populates a DeviceGemmBasic::Arguments structure from the given commandline options

//从给定的命令行选项填充DeviceGemmBasic::Arguments结构

/*

args_from_options 分为 DeviceGemmBasic 和 DeviceGemmStreamK 两个版本。
根据 Options 构造出 GemmUniversal::Arguments，即 GemmUniversalBase::Arguments，即 GemmUniversal::Arguments。

GemmUniversal::Arugments 初始化方法包括：
    Arguments():
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr),
      ptr_gather_A_indices(nullptr),
      ptr_gather_B_indices(nullptr),
      ptr_scatter_D_indices(nullptr)
    {}

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride stride_a,
      typename LayoutB::Stride stride_b,
      typename LayoutC::Stride stride_c,
      typename LayoutC::Stride stride_d,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      stride_a(stride_a), stride_b(stride_b), stride_c(stride_c), stride_d(stride_d),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices)
*/
typename DeviceGemmBasic::Arguments args_from_options(
    const DeviceGemmBasic &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA,LayoutA> tensor_a,
    cutlass::HostTensor<ElementB,LayoutB> tensor_b,
    cutlass::HostTensor<ElementC,LayoutC> tensor_c,
    cutlass::HostTensor<ElementC,LayoutC> tensor_d)
{
    return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversal::kGemm, // universal mode
        options.problem_size, // problem_size
        options.split_k_factor, // batch count / splitk slices
        { // epilogue parameters
            ElementAccumulator(options.alpha),
            ElementAccumulator(options.beta),
        },
        tensor_a.device_data(), // ptr_A
        tensor_b.device_data(), // ptr_B
        tensor_c.device_data(), // ptr_C
        tensor_d.device_data(), // ptr_D
        options.problem_size.mk().product(), // batch_stride_A
        options.problem_size.nk().product(), // batch_stride_B
        options.problem_size.mn().product(), // batch_stride_C
        options.problem_size.mn().product(), // batch_stride_D
        tensor_a.layout().stride(0), // stride_a
        tensor_b.layout().stride(0), // stride_b
        tensor_c.layout().stride(0), // stride_c
        tensor_d.layout().stride(0)); // stride_d    
}

/// Populates a DeviceGemmStreamK::Arguments structure from the given commandline options
typename DeviceGemmStreamK::Arguments args_from_options(
    const DeviceGemmBasic &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA,LayoutA> tensor_a,
    cutlass::HostTensor<ElementB,LayoutB> tensor_b,
    cutlass::HostTensor<ElementC,LayoutC> tensor_c,
    cutlass::HostTensor<ElementC,LayoutC> tensor_d)
{
    return typename DeviceGemmStreamK::Arguments(
        cutlass::gemm::GemmUniversal::kGemm, // universal mode
        options.problem_size, // problem_size
        options.split_k_factor, // batch count / splitk slices
        { // epilogue parameters
            ElementAccumulator(options.alpha),
            ElementAccumulator(options.beta),
        },
        tensor_a.device_data(), // ptr_A
        tensor_b.device_data(), // ptr_B
        tensor_c.device_data(), // ptr_C
        tensor_d.device_data(), // ptr_D
        options.problem_size.mk().product(), // batch_stride_A
        options.problem_size.nk().product(), // batch_stride_B
        options.problem_size.mn().product(), // batch_stride_C
        options.problem_size.mn().product(), // batch_stride_D
        tensor_a.layout().stride(0), // stride_a
        tensor_b.layout().stride(0), // stride_b
        tensor_c.layout().stride(0), // stride_c
        tensor_d.layout().stride(0), // stride_d
        options.avail_sms); // avail_sms
}

/// Execute a given example GEMM computation
template <typename DeviceGemmT>
Result run(std::string description,Options &options){
    // Display test description
    std::cout << std::endl << description << std::endl;

    // Zero-initialize test output matrix D
    /*
    TensorFill 用标量元素填充张量。
    */
    cutlass::reference::host::TensorFill(options.tensor_d.host_view);
    options.tensor_d.sync_device();

    // Instantiate CUTLASS kernel depending on templates
    /*
    创建一个 GemmUniversal 对象。
    */
    DeviceGemmT device_gemm,

    // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
    auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);

    /*
    GemmUniversalBase::get_workspace_size 返回由这些参数表示的问题几何形状所需的工作区大小（以字节为单位）。
    */
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);

    /*
    allocation 即 DeviceAllocation。构造函数调用 allocate 申请内存。
    */
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check the problem size is supported or not
    /*
    GemmUniversalBase::can_implement 判断能否实现： grid 是否超出以及形状是否满足对齐要求。
    */
    CUTLASS_CHECK(device_gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    /*
    GemmUniversalBase::initialize 初始化参数。
    */
    CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    /*
    进行功能测试。
    调用不带入参的 GemmUniversalBase::operator() 函数。
    */
    CUTLASS_CHECK(device_gemm());

    // Copy output data from CUTLASS and reference kernel to host for comparison
    options.tensor_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    Result result;
    /*
    TensorEquals 检查输出是否和参考值的每个元素都相等。
    */
    result.passed = cutlass::reference::host::TensorEquals(
        options.tensor_d.host_view(),
        options.tensor_ref_d.host_view());
    
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    // Run profiling loop
    if(options.iterations > 0){
        /*
        性能测试。
        GpuTimer 通过 cudaEvent 计时。
        gflops 为实际计算吞吐量。
        */
        GPUTimer timer;
        timer.start();
        for(int iter=0; iter<options.iterations; ++iter){
            CUTLASS_CHECK(device_gemm());
        }
        timer.stop();

        // Compute average runtime and GFLOPs.
        float elapsed_millis = timer.elapsed_millis();
        result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
        result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

        std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
        std::cout << "  GFLOPs: " << result.gflops << std::endl;
    }

    if (!result.passed) {
        exit(-1);
    }

    return result;
}

/// Program entrypoint
int main(int argc, const char **argv)
{
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    /*
    检查 CUDA Toolkit 版本。
    */
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }
    // Current device must must have compute capability at least 80
    cudaDeviceProp props;
    int current_device_id;

    /*
    cudaGetDevice 为 CUDA Runtime API，返回当前正在使用的设备。
    */
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    /*
    cudaGetDeviceProperties 返回有关计算设备的信息 cudaDeviceProp 。
    */
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
    /*
    检查设备计算能力。这里要求 SM80以上。
    */
    if (!((props.major * 10 + props.minor) >= 80))
    {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;

        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    // Parse commandline options
    Options options("ampere_streamk_gemm");
    options.parse(argc,argv);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }
    
    std::cout <<
        options.iterations << " timing iterations of " <<
        options.problem_size.m() << " x " <<
        options.problem_size.n() << " x " <<
        options.problem_size.k() << " matrix-matrix multiply" << std::endl;
    
    if (!options.valid()) {
        std::cerr << "Invalid problem." << std::endl;
        return -1;
    }

    //
    // Initialize GEMM datasets
    //

    // Initialize tensors using CUTLASS helper functions
    /*
    HostTensor::resize 改变逻辑张量的大小。
    */
    options.tensor_a.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
    options.tensor_b.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
    options.tensor_c.resize(options.problem_size.mn());       // <- Create matrix C with dimensions M x N
    options.tensor_d.resize(options.problem_size.mn());       // <- Create matrix D with dimensions M x N used to store output from CUTLASS kernel
    options.tensor_ref_d.resize(options.problem_size.mn());   // <- Create matrix D with dimensions M x N used to store output from reference kernel

    // Fill matrix A on host with uniform-random data [-2, 2]
    /*
    TensorFillRandomUniform 函数通过 std::rand 生成随机数。
    */
    cutlass::reference::host::TensorFillRandomUniform(
        options.tensor_a.host_view(), //HostTensor::host_view 返回一个 TensorView 对象。
        1,
        ElementA(2),
        Element(-2),
        0);
    
    // Fill matrix B on host with uniform-random data [-2, 2]
    cutlass::reference::host::TensorFillRandomUniform(
        options.tensor_b.host_view(),
        1,
        ElementB(2),
        Element(-2),
        0);
    
    // Fill matrix C on host with uniform-random data [-2, 2]
    cutlass::reference::host::TensorFillRandomUniform(
        options.tensor_c.host_view(),
        1,
        ElementC(2),
        ElementC(-2),
        0);
    
    //
    // Compute reference output
    //

    // Copy data from host to GPU
    /*
    HostTensor::sync_device 拷贝数据到设备端。
    */
    options.tensor_a.sync_device();
    options.tensor_b.sync_device();
    options.tensor_c.sync_device();

    // Zero-initialize reference output matrix D
    cutlass::reference::host::TensorFill(options.tensor_ref_d.host_view());
    options.tensor_ref_d.sync_device();

    // Create instantiation for device reference gemm kernel
    /*
    DeviceGemmReference 即 Gemm（cutlass::reference::device::Gemm）。调用参考 kernel 的计算结果。
    */
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel 
    
    gemm_reference(
        options.problem_size,
        ElementAccumulator(alpha),
        options.tensor_a.device_ref(), //HostTensor::device_ref 返回一个 TensorRef 对象。
        options.tensor_b.device_ref(), // TensorRef: pointer and layout object referencing a tensor
        ElementAccumulator(beta),
        options.tensor_c.device_ref(),
        options.tensor_ref_d.device_ref());
    
    // Wait for kernels to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output data from reference kernel to host for comparison
    /*
    HostTensor::sync_host 拷贝数据到主机端。
    */
    options.tensor_ref_d.sync_host();

    //
    // Evaluate CUTLASS kernels
    //

    // Test default operation
    if(options.split_k_factor == 1){ // options.split_k_factor=1时比较 Basic-DP 和 StreamK。
        // Compare basic data-parallel version versus StreamK version using default load-balancing heuristics
        /*      
        调用 run 模板函数来运行参数实例化的 kernel: DeviceGemmBasic 和 DeviceGemmStreamK 均为 GemmUniversal。
        前者使用 GemmIdentityThreadblockSwizzle; 后者使用 ThreadblockSwizzleStreamK。
        */
        Result basic_dp = run<DeviceGemmBasic>("Basic data-parallel GEMM", options);
        Result streamk_default = run<DeviceGemmStreamK>("StreamK GEMM with default load-balancing", options);

        printf("  Speedup vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / streamk_default.avg_runtime_ms));

        // Show that StreamK can emulate basic data-parallel GEMM when we set the number of SMs to load-balance across = 1
        options.avail_sms = 1; // Set loadbalancing width to 1 SM (no load balancing)
        Result streamk_dp = run<DeviceGemmStreamK>("StreamK emulating basic data-parallel GEMM", options);
        options.avail_sms = -1; // Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)

        printf("  Speedup vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / streamk_dp.avg_runtime_ms));
        
        // options.split_k_factor自增
        options.split_k_factor++;     // Increment splitting factor for next evaluation
    }

    // Show that StreamK can emulate "Split-K" with a tile-splitting factor
    Result basic_splitk = run<DeviceGemmBasic>(
        std::string("StreamK emulating Split-K GEMM with tile-splitting factor ") + std::to_string(options.split_k_factor),
        options);

    Result streamk_splitk = run<DeviceGemmStreamK>(
        std::string("StreamK emulating Split-K GEMM with tile-splitting factor ") + std::to_string(options.split_k_factor),
        options);
    
    printf("  Speedup vs Basic-SplitK: %.3f\n", (basic_splitk.avg_runtime_ms / streamk_splitk.avg_runtime_ms));

    return 0;
}






