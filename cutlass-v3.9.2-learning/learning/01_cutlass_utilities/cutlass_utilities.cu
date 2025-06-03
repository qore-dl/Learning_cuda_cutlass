/*
  This example demonstrates several CUTLASS utilities in the context of a mixed-precision floating-point matrix product computation.

  这个示例演示了混合精度浮点矩阵乘积计算上下文中的几个CUTLASS实用程序。

  These utilities are intended to be useful supporting components for managing tensor and matrix memory allocations, 
  initializing and comparing results, and computing reference output.

  这些工具旨在作为管理张量和矩阵内存分配、初始化和比较结果以及计算参考输出的有用辅助组件。

  CUTLASS utilities are defined in the directory `tools/util`, and definitions appear namespace `cutlass::` or an inner namespace therein. 
  CUTLASS 的相关实用工具定义在“tools/util”目录中，其定义内容出现在“cutlass:：”命名空间内或该命名空间下的内部命名空间中。
  Operations in `cutlass::reference::` have both host-side and device-side implementations, 
  and the choice to use device-side initialization and host-side verification in this example was arbitrary.
  在 `cutlass::reference::` 中的操作既有主机端的实现也有设备端的实现，而在本示例中选择使用设备端的初始化和主机端的验证这一做法是任意决定的。


  cutlass::half_t

    This is a numeric type implementing IEEE half-precision quantities. It is functional in host and device code. 
    In host-side code, CUTLASS_ENABLE_F16C optionally enables harware-accelerated numeric conversion on x86-64 CPUs support F16C extensions. 
    In device code, all available hardware is used to implement conversion and numeric operations.

  cutlass::half_t
  这是一个实现 IEEE 半精度量的数值类型。它在主机端和设备端代码中均可使用。
  在主机端代码中，CUTLASS_ENABLE_F16C 选项可启用支持 F16C 扩展的 x86-64 CPU 上的硬件加速数值转换。
  在设备端代码中，将使用所有可用硬件来实现数值转换和数值运算。

  cutlass::HostTensor<>

    This template class simplifies the creation of tensors for all supported layouts. 
    It simplifies allocation and management of host- and device- memory allocations.

    This class offers methods device_view() and host_view() to provide TensorView objects for device- and host-side memory allocations.
  
  cutlass::HostTensor<>
  此模板类简化了所有支持布局的张量的创建。
  它简化了主机内存和设备内存分配的分配与管理。
  此类提供了 device_view() 和 host_view() 方法，用于为设备端和主机端的内存分配提供 TensorView 对象。

  cutlass::reference::device::TensorFillRandomGaussian()

    This template function initializes elementsof a tensor to a random Gaussian distribution. 
    It uses cuRAND in device code to compute random numbers.

  cutlass::reference::device::TensorFillRandomGaussian()
  此模板函数将张量的元素初始化为随机高斯分布。
  它在设备代码中使用 cuRAND 来计算随机数。


  cutlass::reference::host::Gemm<>

    This template function computes the general matrix product. 
    This template supports unique data types for each matrix operand, the internal accumulation type, and the scalar parameters alpha and beta.
  
  cutlass::reference::host::Gemm<>
  此模板函数用于计算一般矩阵乘法。
  此模板支持为每个矩阵操作数指定独特的数据类型、内部累加类型以及标量参数 alpha 和 beta 。

  cutlass::reference::host::TensorEquals()

    Compares two tensors of identical rank and returns true if values are bit equivalent.

  cutlass::reference::host::TensorEquals()
  比较两个秩相同的张量，如果值是位相等则返回true。
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

// CUTLASS includes needed for half-precision GEMM kernel
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

//
// CUTLASS utility includes
// CUTLASS 应用所需的include
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"
// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"
// Defines cutlass::half_t
#include "cutlass/numeric_types.h"

// Defines device_memory::copy_device_to_device()
#include "cutlass/util/device_memory.h"

// Defines cutlass::reference::device::TensorFillRandomGaussian()
#include "cutlass/util/reference/device/tensor_fill.h"

// Defines cutlass::reference::host::TensorEquals()
#include "cutlass/util/reference/host/tensor_compare.h"

// Defines cutlass::reference::host::Gemm()
#include "cutlass/util/reference/host/gemm.h"

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t cutlass_hgemm_nn(
  int M,
  int N,
  int K,
  cutlass::half_t alpha,
  cutlass::half_t const *A,
  cutlass::layout::ColumnMajor::Stride::Index lda,
  cutlass::half_t const *B,
  cutlass::layout::ColumnMajor::Stride::Index ldb,
  cutlass::half_t beta,
  cutlass::half_t *C,
  cutlass::layout::ColumnMajor::Stride::Index ldc) {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, // Element A
    cutlass::layout::ColumnMajor, // Layout A
    cutlass::half_t, // Element B
    cutlass::layout::ColumnMajor, // LayoutB
    cutlass::half_t, // Element Output
    cutlass::layout::ColumnMajor>; // Layout Output
  
  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {M,N,K}, //Size of matrix
    {A,lda}, // matrix A with stride
    {B,ldb}, //matrix B with stride
    {C,ldc},
    {C,ldc}, // matrix output with stride
    {alpha,beta} // scalar parameter
  });

  if(status != cutlass::Status::kSuccess){
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.

/*
在 GPU 显存中分配数个矩阵，并调用一个单精度的 CUTLASS GEMM 内核
*/

cudaError_t TestCutlassGemm(int M,int N,int K,cutlass::half_t alpha,cutlass::half_t beta){
  cudaError_t result;
  //
  // Construct cutlass::HostTensor<> using the half-precision host-side type.
  //
  // cutlass::HostTensor<> allocates memory on both the host and device corresponding to rank=2
  // tensors in column-major layout. Explicit synchronization methods are offered to copy the tensor to the device or to the host.
  //
  /*
  使用单精度主机端类型来构建 cutlass::HostTensor<> 实例
  cutlass::HostTensor<> 在主机和设备内存上同时分配rank=2的张量。张量为column-major(列主序)布局。
  提供了显式的同步方法来将tensor复制到设备或主机
  */
  /// M-by-K matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t,cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M,K));

  // K-by-N matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t,cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K,N));

  // M-by-N matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t,cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M,N));

  // M-by-N matrix of cutlass::half_t
  cutlass::HostTensor<cutlass::half_t,cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M,N));

  //
  // Initialize matrices with small, random integers.
  //

  // Arbitrary RNG seed value. Hard-coded for deterministic results.
  uint64_t seed = 2080;

  //Gaussion random distribution
  cutlass::half_t mean = 0.0_hf;
  cutlass::half_t stddev = 5.0_hf;

  // Specify the number of bits right of the binary decimal that are permitted to be non-zero. 指定二进制小数点右边允许非零的位数。
  // A value of "0" here truncates random values to integers
  int bits_less_than_one = 0;

  cutlass::reference::device::TensorFillRandomGaussian(
    A.device_view(),
    seed,
    mean,
    stddev,
    bits_less_than_one);
  
  cutlass::reference::device::TensorFillRandomGaussian(
    B.device_view(),
    seed * 2019,
    mean,
    stddev,
    bits_less_than_one);
    
  cutlass::reference::device::TensorFillRandomGaussian(
    C_cutlass.device_view(),
    seed * 1993,
    mean,
    stddev,
    bits_less_than_one);
  
  // Copy C_cutlass into C_reference so the GEMM is correct when beta != 0.
  //Copy C_cutlass into C_reference (device to device)
  cutlass::device_memory::copy_device_to_device(
    C_reference.device_data(),
    C_cutlass.device_data(),
    C_cutlass.capacity());
  
  // Copy the device-side view into host memory
  C_reference.sync_host();

   //
  // Launch the CUTLASS GEMM kernel
  //
  result = cutlass_hgemm_nn(
    M,
    N,
    K,
    alpha,
    A.device_data(),
    A.stride(0),
    B.device_data(),
    B.stride(0),
    beta,
    C_cutlass.device_data(),
    C_cutlass.stride(0));
  
  if (result != cudaSuccess) {
    return result;
  }

   //
  // Verify the result using a host-side reference
  //

  // A and B were initialized using device-side procedures. The intent of this example is to use the host-side reference GEMM, 
  // so we must perform a device-to-host copy.
  /*
  A 和 B使用设备端过程来初始化
  本例的目的是使用主机端参考GEMM验证cutlass 矩阵乘法，所以必须进行device-to-host 乘法
  */

  A.sync_host();
  B.sync_host();

  // Copy CUTLASS's GEMM results into host memory.
  C_cutlass.sync_host();

  // Compute the reference result using the host-side GEMM reference implementation.
  cutlass::reference::host::Gemm<
    cutlass::half_t, // Element A
    cutlass::layout::ColumnMajor, // Layout A
    cutlass::half_t, // Element B
    cutlass::layout::ColumnMajor, // Layout B
    cutlass::half_t, // Element Output
    cutlass::layout::ColumnMajor, // Layout Output
    cutlass::half_t,
    cutlass::half_t> gemm_ref;
  
  gemm_ref(
    {M,N,K},  // problem size (type: cutlass::gemm::GemmCoord)
    alpha,   // alpha (type: cutlass::half_t)
    A.host_ref(), // A (type: TensorRef<half_t, ColumnMajor>)
    B.host_ref(), // B (type: TensorRef<half_t, ColumnMajor>)
    beta, // beta (type: cutlass::half_t)
    C_reference.host_ref()); // C (type: TensorRef<half_t,ColumnMajor>)
  
  // Compare reference to computed results.
  /*
  cutlass device_view(), device_data()
  device_view():
device_view() 通常返回一个视图对象，这个视图对象包含了对设备上数据的引用以及相关的元数据（如形状、步幅等）。
这个视图对象可以用于在设备上执行各种操作，而不需要将数据显式地复制到主机（CPU）内存中。
视图的一个重要特性是，它不拥有数据，而只是对数据的一个引用。因此，使用视图不会导致数据的复制。
device_data():
device_data() 返回的是一个指向设备上实际数据的指针。
这个指针可以用于直接访问和操作设备内存中的数据。
使用 device_data() 时，开发者需要对设备内存管理有一定的了解，因为直接操作指针可能导致内存泄漏或非法访问等问题。

总结来说，device_view() 提供了一种更高层次的抽象，便于在设备上进行复杂操作时管理数据，而 device_data() 则提供了更底层的访问方式，适合需要直接操作设备内存的场景。
选择使用哪一个取决于具体的应用需求和开发者的熟悉程度。
  */
  if(!cutlass::reference::host::TensorEquals(
    C_reference.host_view(),
    C_cutlass.host_view()
  )){
    char const *filename = "errors_01_cutlass_utilities.csv";

    std::cerr << "Error - CUTLASS GEMM kernel differs from reference. Wrote computed and reference results to '" << filename << "'" << std::endl;

    //
    // On error, print C_cutlass and C_reference to std::cerr.
    //
    // Note, these are matrices of half-precision elements stored in host memory as
    // arrays of type cutlass::half_t.
    //

    std::ofstream file(filename);

    // Result of CUTLASS GEMM kernel
    file << "\n\nCUTLASS =\n" << C_cutlass.host_view() << std::endl;
    
    // Result of reference computation
    file << "\n\nReference =\n" << C_reference.host_view() << std::endl;

    // Return error code.
    return cudaErrorUnknown;
  }

  // Passed error check
  return cudaSuccess;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to cutlass_utilities example.
//
// usage:
//
//   01_cutlass_utilities <M> <N> <K> <alpha> <beta>
//

int main(int argc,char *argv[]){
  //
  // This example uses half-precision and is only suitable for devices with compute capabitliy 5.3 or greater.
  //

  cudaDeviceProp prop;
  cudaError_t result = cudaGetDeviceProperties(&prop,0);

  if (result != cudaSuccess) {
    std::cerr << "Failed to query device properties with error " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  if(!(prop.major>5||(prop.major==5 && prop.minor >= 3))){
    std::cerr << "This example uses half precision and is only suitable for devices with compute capability 5.3 or greater.\n";
    std::cerr << "You are using a CUDA device with compute capability " << prop.major << "." << prop.minor << std::endl;
    return -1;
  }

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions: <M> <N> <K>
  int problem[3] = {128,128,128};

  for (int i = 1; i < argc && i < 4; ++i){
    std::stringstream ss(argv[i]);
    ss >> problem[i-1];
  }

  // Linear scale factors in GEMM. Note, these are half-precision values stored as
  // cutlass::half_t.
  //
  // Values outside the range of IEEE FP16 will overflow to infinity or underflow to zero.
  //
  cutlass::half_t scalars[2] = {1.0_hf,0.0_hf};
  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(argv[i]);

    ss >> scalars[i - 4];   // lexical cast to cutlass::half_t
  }

  //
  // Run the CUTLASS GEMM test.
  //

  result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}