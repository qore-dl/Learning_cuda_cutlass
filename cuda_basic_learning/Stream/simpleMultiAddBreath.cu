#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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
6.3.2 使用广度优先调度重叠
先前的例子表面，当采用广度优先的方式调度内核时，Fermi GPU 可以实现最好的效果。
现在，将在重叠数据传输和计算内核中，检验广度优先排序产生的效果。
下面的代码演示了使用广度优先的方法来调度流间的计算和通信：
*/

int main(int argc,char **argv){
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
    char *iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname,"1",1);
    char *ivalue = getenv(iname);
    printf ("> %s = %s\n", iname, ivalue);
    printf ("> with streams = %d\n", NSTREAM);

    // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *hostRef, *gpuRef;
    cudaHostAlloc((void**)&h_A,nBytes,cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B,nBytes,cudaHostAllocDefault);
    cudaHostAlloc((void**)&gpuRef,nBytes,cudaHostAllocDefault);
    cudaHostAlloc((void**)&gpuRef,nBytes,cudaHostAllocDefault);

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

    for(int i=0;i<NSTREAM;i++){
        cudaStreamCreate(&(stream[i]));
    }

    cudaEventRecord(start,0);
    /*
    下面的代码演示了使用广度优先的方法来调度流间的计算和通信：
    */
    // initiate all asynchronous transfers to the device
    for(int i = 0;i < NSTREAM;i++){
        int ioffset = i *iElem;
        cudaMemcpyAsync(&d_A[ioffset],&h_A[ioffset],iBytes,
                        cudaMemcpyHostToDevice,stream[i]);
        cudaMemcpyAsync(&d_B[ioffset],&h_B[ioffset],iBytes,
                        cudaMemcpyHostToDevice,stream[i]);
    }

    // launch a kernel in each stream
    for(int i=0;i < NSTREAM;i++){
        int ioffset = i *iElem;
        sumArrays<<<grid,block,0,stream[i]>>>(&d_A[ioffset],&d_B[ioffset],
                                            &d_C[ioffset],iElem,iElem);
    }

    // enqueue asynchronous transfers from the device
    for(int i=0;i<NSTREAM;++i){
        int ioffset = i * iElem;
        cudaMemcpyAsync(&gpuRef[ioffset],&d_C[ioffset],iBytes,
                        cudaMemcpyDeviceToHost,stream[i]);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float execution_time;
    cudaEventElapsedTime(&execution_time,start,stop);

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
在使用一个工作队列的情况下，与深度优先方法相比，广度优先方法没有明显的差异：
$./simpleMultiAddDepth 
> ./simpleMultiAddDepth Starting...
> Using Device 0: Tesla V100-SXM2-16GB
> Compute Capability 7.0 hardware with 80 multi-processors
> CUDA_DEVICE_MAX_CONNECTIONS = 1
> with streams = 4
> vector size = 262144
> grid (2048, 1) block (128, 1)

Measured timings (throughput):
 Memcpy host to device  : 0.351040 ms (2.987056 GB/s)
 Memcpy device to host  : 0.179904 ms (5.828531 GB/s)
 Kernel                 : 214.793442 ms (0.009764 GB/s)
 Total                  : 215.324387 ms (0.009740 GB/s)

Actual results from overlapped data transfers:
 overlap with 4 streams : 52.049343 ms (0.040292 GB/s)
 speedup                : 75.827469 
Arrays match.

$./simpleMultiAddBreath 
> ./simpleMultiAddBreath Starting...
> Using Device 0: Tesla V100-SXM2-16GB
> Compute Capability 7.0 hardware with 80 multi-processors
> CUDA_DEVICE_MAX_CONNECTIONS = 1
> with streams = 4
> vector size = 262144
> grid (2048, 1) block (128, 1)

Measured timings (throughput):
 Memcpy host to device  : 0.344032 ms (3.047903 GB/s)
 Memcpy device to host  : 0.175456 ms (5.976290 GB/s)
 Kernel                 : 227.794876 ms (0.009206 GB/s)
 Total                  : 228.314362 ms (0.009185 GB/s)

Actual results from overlapped data transfers:
 overlap with 4 streams : 47.245441 ms (0.044388 GB/s)
 speedup                : 79.306847 
Arrays match.
因为 GPU 中的双向调度机制有助于消除虚假的依赖关系。

但如果在 Fermi 设备上运行相同的测试，在整体性能方面会发现，使用广度优先方法不如使用
深度优先方法。由主机到设备复制队列上的争用导致虚假的依赖关系，
在主机到设备间传输完成前，将阻止所有内核的启动

Kepler 以及以后的GPU：因为内存复制数据传输走的是硬件工作队列以外的双向通道，所以只要不存在流内依赖关系不会阻塞硬件工作队列中的kernel运行。

因此，对于 Kepler 及以后的设备而言，在大多数情况下无需罐组其工作调度顺序。
而在 Fermi 设备上，要注意这些问题，并且对不同的调度方案做出评估，使工作负载找到最佳地任务调度顺序。
*/