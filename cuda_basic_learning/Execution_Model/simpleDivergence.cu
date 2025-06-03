/*
线程束分化实验
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
线程束分化实验

1. 当一个分化的线程采取不同的代码路径时，会产生线程束分化
2. 不同的 if-then-else 分支会连续执行
3. 尝试调整分支的粒度以适应线程束大小的倍数可以有效地避免线程束分化
4. 不同的分化可以执行不同的代码，且无须以牺牲性能为代价（线程的状态分别存放于不同的寄存器文件，且线程束的状态已存入共享内存-一级缓存，即状态已加载，上下文切换成本极小）

线程束的本地执行上下文主要由以下资源组成：
程序计数器
寄存器
共享内存
由 SM 处理的每个线程束的执行上下文在整个线程束的生命周期中是保存在芯片内的存储中的，因此从一个执行上下文切换到另一个执行上下文没有损失。
*/

/*
 * simpleDivergence demonstrates divergent code on the GPU and its impact on
 * performance and CUDA metrics.
 */

/*
==111741== Profiling application: ./simpleDivergence
==111741== Profiling result:
==111741== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: mathKernel1(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: mathKernel2(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: mathKernel3(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: mathKernel4(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: warmingup(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
          
kernel1中出现了分支分化现象，但是在nvprof 中 branch_efficiency (即未分化分支的占比)
因为在同一warp内部，必须同时执行相同的指令，遇到分支判断后则会导致复合不同条件的线程运行不同的指令
因此，在出现分支时会有一部分线程被禁用，该现象被称为线程束分化

但是，在kernel1中，因为CUDA编译器的自动优化，短的、有条件的代码段的断定指令取代了分支指令（导致分化的实际控制流指令）

在分支预测时，根据条件，把每个线程中的一个断定变量设置为1或0。这两种条件流路径被完全执行，但只有断定为1的指令被执行
断定为0的指令不被执行，但是相应的线程也不会停止。这和实际的分支指令之间的区别是微妙的
只有在条件语句的指令数小于某个阈值时，编译器才会用断定指令替换分支指令。因此，一段很长的代码路径肯定会导致线程束分化
可以使用以下指令来强制CUDA编译器不利用分支预测去优化kernel：
nvcc -g -G simpleDeivergence.cu -o simpleDivergence

*/
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

/*
kernel2 不报告分支分化的唯一原因：分支粒度是线程束大小的整数倍
*/

__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

/*
kernel3 使内核代码的分支预测结果直接显示：ipred
CUDA对 kernel1 和 kernel3 进行有限的优化，以保持分支效率维持在50%以上，
if ... else ... 分离为kernel3中的多个 if 语句时，分化分支的数量会翻倍
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: mathKernel1(float*)
          1                         branch_efficiency                         Branch Efficiency      85.71%      85.71%      85.71%
    Kernel: mathKernel2(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: mathKernel3(float*)
          1                         branch_efficiency                         Branch Efficiency      80.00%      80.00%      80.00%
    Kernel: mathKernel4(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: warmingup(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
*/

__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 64;
    int blocksize = 64;

    if(argc > 1) blocksize = atoi(argv[1]);

    if(argc > 2) size      = atoi(argv[2]);

    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // run a warmup kernel to remove overhead
    size_t iStart, iElaps;
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("warmup      <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps );
    CHECK(cudaGetLastError());

    // run kernel 1
    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps );
    CHECK(cudaGetLastError());

    // run kernel 2
    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps );
    CHECK(cudaGetLastError());

    // run kernel 3
    iStart = seconds();
    mathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel3 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps);
    CHECK(cudaGetLastError());

    // run kernel 4
    iStart = seconds();
    mathKernel4<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel4 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x,
           iElaps);
    CHECK(cudaGetLastError());

    // free gpu memory and reset divece
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
