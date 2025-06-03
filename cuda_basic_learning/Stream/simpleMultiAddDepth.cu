#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


/*
6.3 重叠内核执行和数据传输
在前一节中，已经介绍了如何在多个流中并发执行多个内核。
在本节中，将学习如何并发执行内核和数据传输。
重叠内核和数据传输表现出不同的行为，并且需要考虑一些并发内核执行相比不同的因素。

Fermi GPU 和 Kepler GPU 有两个复制引擎队列：
一个用于将数据传输到设备，另一个用于从设备中将数据提取出来。
因此，最多可以重叠两个数据传输，并且只有当他们的方向不同
并且被调度到不同的流时才能这样做，否则，所有的数据传输都将是串行的。
在决定如何使用内核计算最佳地重叠数据传输时，记住这一点是很重要的。

在应用程序中，还需要检验数据传输和内核执行之间的关系，从而可以区分以下两种情况：
1. 如果一个内核使用数据 A，那么对 A 进行数据传输必须要安排在内核启动前，
且必须位于相同的流中；
2.如果一个内核完全不使用数据 A，那么内核执行和数据传输可以位于不同的流中。
在第二种情况下，实现内核和数据传输的并发执行是很容易的：
将它们放置在不同的流中，这就已经向runtime表示了并发地执行它们是安全的。
然而，在第一种情况下，要实现数据传输和内核执行之间的重叠会更加复杂，
因为内核依赖数据作为输入。当内核和传输之间存在依赖关系时，
可以使用向量加法示例来检验如何实现重叠数据传输和内核执行。
*/

#define NSTREAM 4
#define BDIM 128

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

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
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}

/*
6.3.1 使用深度优先调度重叠
*/

__global__ void sumArrays(float *A, float *B, float *C, const int N,const int n_repeat)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        for (int i = 0; i < n_repeat; ++i)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
}

/*
本节中，唯一增加的变动就是这个内核计算被n_repest增强了，以延长内核的执行时间。

实现向量加法的 CUDA 程序，其基本结构包含3个主要步骤：
1. 将两个输入向量从主机复制到设备中；
2. 执行向量加法运算
3. 将单一的输出向量从设备返回主机中；

从这些步骤中，也许不能明显看出计算和通信时如何被重叠的。
为了在向量加法中实现重叠，需要将输入和输出数据集划分成子集，
并将来自一个子集的通信与来自其他子集的计算进行重叠。
具体对向量加法来说，需要将两个长度为 N 的向量加法问题划分为
长度为 N/M 的向量相加的 M 个子问题。因为这里的每个子问题都是独立的，
所以每一个被安排在不同的 CUDA 流中，这样它们的计算和通信就可以重叠了。

在第2章的向量加法程序中，数据传输是通过同步复制函数来实现的。
要重叠数据传输和内核执行，必须使用异步复制函数。
因为异步复制函数需要固定的主机内存，所以首先需要使用
cudaHostAlloc函数，固定主机内存中修改主机数组的分配：
cudaHostAlloc((void**)&gpuRef,nBytes,cudaHostAllocDefault);
cudaHostAlloc((void**)&hostRef,nBytes,cudaHostAllocDefault);

接下来，需要在 NSTREAM 个流中，平均分配该问题的任务。每一个流要处理的元素数量
使用如下代码进行定义：
int iElem = nElem / NSTREAM;

现在，可以使用一个循环来为几个流同时调度 iElem 个元素的通信和计算，代码如下：
for(int i=0;i<NSTREAM;i++){
    int offset = i*iElem;
    cudaMemcpyAsync(&d_A[offset],&h_A[offset],iBytes,cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(&d_B[offset],&h_A[offset],iBytes,cudaMemcpyHostToDevice,stream[i]);
    sumArrays<<<grid,block,0,stream[i]>>>(&d_A[offset],&d_B[offset],&d_C[offset],iElem,iElem);
}
由于这些内存复制和内核启动对主机而言是异步的，因此全部的工作负载都可以毫无阻塞的
在流之间进行分配。通过将数据传输和该数据上的计算放置在同一个流中，
输入向量、内核计算以及输出向量之间的依赖关系可以被保持。
为了进行对比，此例子还是用了一个阻塞实现来计算基准性能：
sumArrays<<<grid,block>>>(d_A,d_B,d_C,nElem,nElem);
*/

int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // set up max connectioin
    char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "1", 1);
    char *ivalue =  getenv (iname);
    printf ("> %s = %s\n", iname, ivalue);
    printf ("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *hostRef, *gpuRef;
    // CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
    // CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
    // CHECK(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
    // CHECK(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));
    cudaHostAlloc((void**)&h_A,nBytes,cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B,nBytes,cudaHostAllocDefault);
    cudaHostAlloc((void**)&gpuRef,nBytes,cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostRef,nBytes,cudaHostAllocDefault);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // invoke kernel at host side
    dim3 block (BDIM);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x,
            block.y);

    // sequential operation
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float memcpy_h2d_time;
    CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem,nElem/NSTREAM);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float kernel_time;
    CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float memcpy_d2h_time;
    CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

    printf("\n");
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device\t: %f ms (%f GB/s)\n",
           memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
    printf(" Memcpy device to host\t: %f ms (%f GB/s)\n",
           memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
    printf(" Kernel\t\t\t: %f ms (%f GB/s)\n",
           kernel_time, (nBytes * 2e-6) / kernel_time);
    printf(" Total\t\t\t: %f ms (%f GB/s)\n",
           itotal, (nBytes * 2e-6) / itotal);

    // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaEventRecord(start, 0));

    // initiate all work on the device asynchronously in depth-first order
    // for (int i = 0; i < NSTREAM; ++i)
    // {
    //     int ioffset = i * iElem;
    //     CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes,
    //                           cudaMemcpyHostToDevice, stream[i]));
    //     CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,
    //                           cudaMemcpyHostToDevice, stream[i]));
    //     sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset],
    //             &d_C[ioffset], iElem);
    //     CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,
    //                           cudaMemcpyDeviceToHost, stream[i]));
    // }
    for(int i=0;i<NSTREAM;i++){
        int ioffset = i*iElem;
        cudaMemcpyAsync(&d_A[ioffset],&h_A[ioffset],iBytes,cudaMemcpyHostToDevice,stream[i]);
        cudaMemcpyAsync(&d_B[ioffset],&h_B[ioffset],iBytes,cudaMemcpyHostToDevice,stream[i]);
        sumArrays<<<grid,block,0,stream[i]>>>(&d_A[ioffset],&d_B[ioffset],&d_C[ioffset],iElem,iElem);
        cudaMemcpyAsync(&gpuRef[ioffset],&d_C[ioffset],iBytes,cudaMemcpyDeviceToHost,stream[i]);
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float execution_time;
    CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nBytes * 2e-6) / execution_time );
    printf(" speedup                : %f \n",
           ((itotal - execution_time) * 100.0f) / itotal);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(hostRef));
    CHECK(cudaFreeHost(gpuRef));

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // destroy streams
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamDestroy(stream[i]));
    }

    CHECK(cudaDeviceReset());
    return(0);
}

/*
$./simpleMultiAddDepth 
> ./simpleMultiAddDepth Starting...
> Using Device 0: Tesla V100-SXM2-16GB
> Compute Capability 7.0 hardware with 80 multi-processors
> CUDA_DEVICE_MAX_CONNECTIONS = 1
> with streams = 4
> vector size = 262144
> grid (2048, 1) block (128, 1)

Measured timings (throughput):
 Memcpy host to device  : 0.361056 ms (2.904192 GB/s)
 Memcpy device to host  : 0.177088 ms (5.921214 GB/s)
 Kernel                 : 76.655998 ms (0.027358 GB/s)
 Total                  : 77.194138 ms (0.027167 GB/s)

Actual results from overlapped data transfers:
 overlap with 4 streams : 55.305408 ms (0.037919 GB/s)
 speedup(%)                : 28.355429 
Arrays match.

$./simpleMultiAddDepth 
> ./simpleMultiAddDepth Starting...
> Using Device 0: Tesla V100-SXM2-16GB
> Compute Capability 7.0 hardware with 80 multi-processors
> CUDA_DEVICE_MAX_CONNECTIONS = 1
> with streams = 4
> vector size = 262144
> grid (2048, 1) block (128, 1)

Measured timings (throughput):
 Memcpy host to device  : 0.344480 ms (3.043939 GB/s)
 Memcpy device to host  : 0.171520 ms (6.113433 GB/s)
 Kernel                 : 218.137283 ms (0.009614 GB/s)
 Total                  : 218.653290 ms (0.009591 GB/s)

Actual results from overlapped data transfers:
 overlap with 4 streams : 48.234848 ms (0.043478 GB/s)
 speedup(%)                : 77.940025 
Arrays match

该例子中，有以下三种重叠：
1.不同流中内核的互相重叠
2.内核与其他流中的数据传输重叠
3.在不同流以及不同方向上的数据传输互相重叠

该例子中，有以下两种阻塞行为：
1. 内核被同一流中先前的数据传输所阻塞
2. 从主机到设备的数据传输被同一方向上先前的数据传输所阻塞

虽然从主机到设备的数据传输是在 4 个不同的流中执行的，但是时间轴显示它们是按顺序执行的。
因为实际上，它们是通过相同的复制引擎队列来执行的。
接下来，可以尝试将硬件工作队列的数量减少至一个，然后重新运行，测试其性能。
在这个例子中，1个工作队列和8个工作队列之间没有显著差异。
因为每个流中只执行单一的一个内核，所以减少工作队列的数目并没有增加虚假依赖关系，
同样，现存的虚假依赖关系（由主机到设备的复制队列所引起的）也没有减少。

减少 K40 中工作队列的数目，可以创造一个类似于 Fermi GPU 的环境：
一个工作队列和两个复制队列。
如果在 Fermi GPU 上运行相同的测试，就会发现虚假的依赖关系是确实存在的。
这是由于 Kepler 的工作调度机制导致的，在网格管理单元（Grid Management Unit,GMU）中实现。
GMU 负责对发送到 GPU 中的工作进行管理和排序。通过对 GMU 的分析有助于减少虚假的依赖关系。结果如下：


网格管理单元（GMU）
Kepler 引入一个新的网格管理和调度控制系统，即网格管理单元（GMU）。
GMU 可以暂停新网格的调度，使得网格排队等待且暂停网格直到它们准备好执行，
这样就使runtime变得非常灵活强大，动态并行就是一个很好的例子。

在 Fermi 设备上，网格直接从流队列被传到 CUDA 工作分配器（CUDA Work Distributor, CWD) 中。
在 Kepler 设备上，网格被发送到 GMU 上，GMU 对在 GPU 上执行的网格进行管理和优先级排序。

GMU 创建了多个硬件工作队列，从而减少或消除了虚假的依赖关系。
通过 GMU，流可以作为单独的工作流水线。
即使 GMU 被限制智能创建一个单一的硬件工作队列，根据以上测试结果证实，
通过 GMU 进行的网格依赖性分析也可以帮助消除虚假的依赖关系（硬件工作队列只要资源足够且软件（如流）上没有设置依赖，则不产生依赖，可并发，消除虚假依赖）
*/


