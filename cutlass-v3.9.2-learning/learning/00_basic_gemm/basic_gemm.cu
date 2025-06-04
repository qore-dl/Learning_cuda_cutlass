/*
这个例子演示了如何调用一个CUTLASS gem内核，并提供了一个简单的引用
矩阵乘法核验证其正确性。

CUTLASS gem模板在函数CutlassSgemmNN中实例化。这是核计算
一般矩阵积（GEMM）使用单精度浮点运算和假设
所有矩阵都有列主布局。

threadblock tile的大小选择为128x128x8，这为大型矩阵提供了良好的性能。
有关可用的可调参数的更多说明，请参阅CUTLASS Parallel for All博客文章
在短剑。

https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

除了定义和启动SGEMM内核之外，这个示例没有使用任何其他组件
或CUTLASS内的公用事业。这些实用程序在其他示例中进行了演示
普遍存在于CUTLASS单元测试中。

这个例子故意保持与从cutlass-1.3到basic_gem的例子相似
突出显示过渡到cutlass-2.0所需的最小差异。

Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
// #include "../common/helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//这个函数定义了一个CUTLASS gem内核实例化，构造了它的参数对象，并在CUDA设备上启动它。
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
    int M, 
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc
){
    // Define type definition for single-precision CUTLASS GEMM with column-major
    // input matrices and 128x128x8 threadblock tile size (chosen by default).
    //
    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as
    // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
    //
    // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`
  /*
  下面进行一些类型的定义，这些定义用来实现 单精度（16位）CUTLASS GEMM（矩阵乘法），该乘法采用列主序输入矩阵
  threadblock tile size: 128x128x8

  为了保持接口的可管理性，为合理的组合定义了几个帮助程序，包括以下单精度GEMM示例。典型值被用作默认模板参数。
  详细信息请参见‘ cultlass /gemm/device/default_gemm_configuration.h ’。
  */

    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, // Data-type of A matrix
        ColumnMajor, // Layout of A matrix
        float, // Data-type pf B matrix
        ColumnMajor, // Layout of B matrix
        float, // Data-type of C matrix
        ColumnMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    /*
    定义一个 CUTLASS GEMM 类型的op
    */

    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    // 
    //  The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.

    /*
    构建一个 CUTLASS-GEMM op的参数实体
    CUTLASS 的一个关键设计模式就是定义在主机端代码中可以任意构建的参数模版实体，之后在具体调用kernel时传入实际的值。
    这些参数可以包括指针、步长、标量以及其他在矩阵乘法及其组件中需要的参数。

    这一模式的优点在于：
    1. 一种结构化、可组合的策略，用于向具体kernel传入在主机端可任意构建配置的参数
    2. 最小化kernel 入口的初始化开销。
    */

    CutlassGemm::Arguments args({M,N,K}, // GEMM problem dimensions
                                {A,lda}, // Tensor-ref(矩阵占位) for source matrix A
                                {B,ldb}, // Tensor-ref for source matrix B
                                {C,ldc}, // Tensor-ref for source matrix C,
                                {C,ldc}, // Tensor-ref for destination matrix D (mab be different memory than source C matrix)
                                {alpha,beta}); //Scalars used in Epilogue
                                

    /*
    Launch the Cutlass GEMM Kernel 发布 GEMM kernel
    */

    cutlass::Status status = gemm_operator(args);

    if(status != cutlass::Status::kSuccess){
        // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
        return cudaErrorUnknown;
    }else{
        // Return success, if no errors were encountered.
        return cudaSuccess;
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/*
下面的代码基本上是通用的 CUDA 代码，使用 CUDA Runtime API 和简单的 CUDA kernels 来初始化矩阵，计算广义矩阵乘法。
*/

/// Kernel to initialize a matrix with small integers.

__global__ void InitializeMatrix_kernel(
    float *matrix,
    int rows,
    int columns,
    int seed = 0
    ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < rows && j << columns){
        int offset = i + j*rows;

        // Generate arbitrary elements.
        int const k = 16807;
        int const m = 16;
        float value = float(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix,int rows,int columns,int seed=0){
    dim3 block(16,16);
    dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);

    InitializeMatrix_kernel<<<grid,block>>>(matrix,rows,columns,seed);
    return cudaGetLastError();

}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix,int rows,int columns,int seed=0){
    cudaError_t result;

    size_t sizeof_matrix = sizeof(float)*rows*columns;

    result = cudaMalloc(reinterpret_cast<void **>(matrix),sizeof_matrix);
    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: "
          << cudaGetErrorString(result) << std::endl;
        return result;
    }
    
    // Clear the allocation.
    result = cudaMemset(*matrix,0,sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: "
          << cudaGetErrorString(result) << std::endl;
        return result;
    }

    result = InitializeMatrix(*matrix,rows,columns,seed);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: "
          << cudaGetErrorString(result) << std::endl;
        return result;
    }
    
    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.

/*
使用cuda core 实现的基本的矩阵乘法
*/

__global__ void ReferenceGemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc
){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    if (i < M && j < N){
        float accumulator = 0;

        for(int k = 0; k < K;k++){
            accumulator += A[i+lda*k] * B[k + j*ldb];
        }

        C[i+j*ldc] = alpha * accumulator + beta * C[i+j*ldc];
    }
}


cudaError_t ReferenceGemm(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc
){
    dim3 block(16, 16);
    dim3 grid(
        (M + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    ReferenceGemm_kernel<<<grid,block>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);

    return cudaGetLastError();
}

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M,int N,int K,float alpha,float beta){
    cudaError_t result;

    //
    // Define several matrices to be used as operands to GEMM kernels.
    //
    // Compute leading dimensions for each matrix.

    // ld: leading dimensions for a matrix
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Compute size in bytes of the C matrix.
    size_t sizeof_C = sizeof(float)*ldc*N;
    // Define pointers to matrices in GPU device memory.
    float *A;
    float *B;
    float *C_cutlass;
    float *C_reference;

    //
    // Allocate matrices in GPU device memory with arbitrary seeds.
    //

    result = AllocateMatrix(&A, M, K, 0);

    if (result !=  cudaSuccess) {
        return result;
    }

    result = AllocateMatrix(&B, K, N, 17);
    
    if (result !=  cudaSuccess) {
        cudaFree(A);
        return result;
    }

    result = AllocateMatrix(&C_cutlass,M,N,101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        return result;
      }
    
    result = AllocateMatrix(&C_reference, M, N, 101);
    
    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C_cutlass);
        return result;
    }

    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
        << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Launch CUTLASS GEMM.
    //

    result = CutlassSgemmNN(M,N,K,alpha,A,lda,B,ldb,beta,C_cutlass,ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
          << cudaGetErrorString(result) << std::endl;
    
        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);
    
        return result;
    }

    //
    // Verify.
    //

    // Launch reference GEMM
    result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
    if (result != cudaSuccess) {
        std::cerr << "Reference GEMM kernel failed: "
            << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    // Copy to host and verify equivalence.
    std::vector<float> host_cutlass(ldc*N,0);
    std::vector<float> host_reference(ldc*N,0);

    result = cudaMemcpy(host_cutlass.data(),C_cutlass,sizeof_C,cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: "
          << cudaGetErrorString(result) << std::endl;
    
        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);
    
        return result;
    }
    
    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);
    
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy Reference GEMM results: "
          << cudaGetErrorString(result) << std::endl;
    
        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);
    
        return result;
    }

    //
    // Free device memory allocations.
    //
    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(A);
    cudaFree(B);

     //
    // Test for bit equivalence of results.
    //
    if (host_cutlass != host_reference) {
        std::cerr << "CUTLASS results incorrect." << std::endl;
        return cudaErrorUnknown;
    }

    return cudaSuccess;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//

int main(int argc, const char *arg[]) {

    //
    // Parse the command line to obtain GEMM dimensions and scalar values.
    //
  
    // GEMM problem dimensions.
    int problem[3] = { 128, 128, 128 };
  
    for (int i = 1; i < argc && i < 4; ++i) {
      std::stringstream ss(arg[i]);
      ss >> problem[i - 1];
    }
  
    // Scalars used for linear scaling the result of the matrix product.
    float scalars[2] = { 1, 0 };
  
    for (int i = 4; i < argc && i < 6; ++i) {
      std::stringstream ss(arg[i]);
      ss >> scalars[i - 4];
    }
  
    //
    // Run the CUTLASS GEMM test.
    //
  
    cudaError_t result = TestCutlassGemm(
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
///////////////////////////////////////////////////////////////////////////////////////////////////