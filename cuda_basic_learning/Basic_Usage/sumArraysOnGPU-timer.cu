#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{   
    //kernel函数内均可被访问到，且为uint3类型。
    //手动定义的dim3类型的grid和block变量仅在host端可见
    //unint3类型的内置预初始化的grid，block和thread变量仅在device端可见
    int i = blockIdx.x * blockDim.x + threadIdx.x; //数据块、线程唯一标识
    //blockIdx.x: 2Dgrid网格中，block在grid的x轴上的编号。
    //blockDim.x: 每个Block组织为3D线程块，即block在X轴上的长度/规模/尺寸
    //threadIdx.x: 每个Block内部，线程的编号
    /*
    CUDA编程中，blockDim.x 和 blockIdx.x 是两个非常重要的内置变量，它们用于标识线程和线程块的组织方式。理解这些变量的含义对于正确编写并行计算的CUDA核函数(kernel)至关重要。
    blockDim.x
    含义：blockDim.x 表示每个线程块（block）中的线程数量，在x维度上。在定义一个线程块时，你可以指定其在x、y、z维度上的尺寸（尽管大多数情况下只使用x维度）。blockDim.x 对应于你在启动kernel时指定的线程块大小。
    用途：通常用于计算当前线程在整个线程块中的相对位置，或者当你需要将工作分配给线程块内的每个线程时。
    
    blockIdx.x
    含义：blockIdx.x 表示当前线程块在其所属网格（grid）中的索引，在x维度上。同样地，网格可以被组织成一维、二维或三维结构，但最常见的是使用一维网格。
    用途：用于识别当前线程块在整个网格中的位置，这在你需要将数据分布到多个线程块进行处理时特别有用。
    
    结合使用
    通常，结合使用 blockDim.x, blockIdx.x, 和 threadIdx.x 可以计算出每个线程的全局唯一ID，从而允许你为每个线程分配不同的任务或数据段。例如：
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    这段代码计算了当前线程在整个网格中的全局索引。假设你有一个一维的数据集，并希望每个线程处理一个元素，那么这个公式可以帮助你确定哪个线程应该处理哪个数据元素。
    */

    /*
    由一个kernel所产生的所有线程统称为一个grid，同一grid中的所有线程共享相同的全局内存空间（gloabl memory）。
    一个grid由多个线程块block构成，一个block包含一组线程，同一block中的线程协作方式包括：
    （1） 同步
    （2） 共享内存（shared memory）
    不同block内的线程不能协作。
    线程依靠以下两个坐标变量来进行区分：
    （1）blockIdx：block在grid中的索引/坐标
    （2）threadIdx: 具体thread在block内的索引/坐标
    这些变量是核函数中需要预初始化的内置变量。
    当执行一个核函数时，CUDA runtime为每个thread分配坐标变量blockIdx和threadIdx。
    基于这些坐标，可以将数据分配给不同的线程。
    */

    /*
    坐标变量是基于uint3定义的CUDA内置的向量类型，是一个包含3个无符号整数的结构，可以通过x,y,z三个字段来指定：
    blockIdx.x
    blockIdx.y
    blockIdx.z
    threadIdx.x
    threadIdx.y
    threadIdx.z

    CUDA可组织三维的grid和block。grid和block的维度由下列两个内置变量指定：
    blockDim: block的维度，用每个block中的thread 数量来表示
    gridDim: grid的维度，用每个grid中的block 数量来表示

    dim3类型的变量，是基于uint3定义的整数型向量，用来表示维度。当定义一个dim3类型的变量时，
    所有未指定的元素都被初始化为1。dim3类型的变量中的每个组件可以通过x/y/z字段获得，如下所示：
    blockDim.x
    blockDim.y
    blockDim.z

    grid和block的维度：
    通常，一个grid会被组织称block的二维数组形式，一个block会被组织成thread的三维数组形式

    grid和block均使用3个dim3类型的无符号整型字段，而未使用的字段将被初始化为1且忽略不计
     */

    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;

    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    //malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A,nBytes));
    CHECK(cudaMalloc((float**)&d_B,nBytes));
    CHECK(cudaMalloc((float**)&d_C,nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C,gpuRef,nBytes,cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = 1024;
    //在主机端，作为kernel调用的一部分，可以使用dim3数据类型定义grid和block的维度
    dim3 block (iLen);
    dim3 grid ((nElem+block.x - 1)/block.x);
    //手动定义的dim3类型的grid和block变量仅在host端可见
    //unint3类型的内置预初始化的grid，block和thread变量仅在device端可见

    iStart = seconds();
    //执行kernel函数时，CUDA runtime会生成相应的内置预初始化的grid、block和thread相关变量（索引，维度等）
    sumArraysOnGPU<<<grid,block>>>(d_A,d_B,d_C,nElem);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    
    // invoke kernel at host side
    iLen = 512;
    //在主机端，作为kernel调用的一部分，可以使用dim3数据类型定义grid和block的维度
    block.x =  iLen;
    grid.x =  (nElem+block.x - 1)/block.x;
    //手动定义的dim3类型的grid和block变量仅在host端可见
    //unint3类型的内置预初始化的grid，block和thread变量仅在device端可见

    iStart = seconds();
    //执行kernel函数时，CUDA runtime会生成相应的内置预初始化的grid、block和thread相关变量（索引，维度等）
    sumArraysOnGPU<<<grid,block>>>(d_A,d_B,d_C,nElem);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);
    
    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

     // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}

/*
了解device自身局限性

在调整执行配置时，需要了解的一个关键点是对grid和block 维度的限制。线程层次结构中每个层级的最大尺寸取决于设备
CUDA提供了通过查询GPU来了解限制的能力

*/