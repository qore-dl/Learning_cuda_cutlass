#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int checkResult(float *data, const int n, const float x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}

/*
6.4 重叠 GPU 和 CPU 执行
相对而言，实现 GPU 和 CPU 执行重叠是比较简单的，
因为所有的内核启动在默认情况下是异步的。
因此，只需要简单地启动内核，并且立即在主机线程上实现有效操作，
就能自动重叠 GPU 和 CPU 执行。
本节的示例主要包括两个部分：
1. 内核被调度到默认流中
2. 等待 GPU 内核时执行主机计算

使用下面的简单内核实现一个向量与标量的加法：
*/

__global__ void kernel(float *g_data,float value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

/*
本例子中使用了3个 CUDA 操作（两个复制和一个内核启动）。记录一个停止事件，
以标记所有 CUDA 操作的完成。
cudaMemcpyAsync(d_a,h_a,nbytes,cudaMemcpyHostToDevice);
kernel<<<grid,block>>>(d_a,value);
cudaMemcpyAsync(h_a,d_a,nbytes,cudaMemcpyDeviceToHost);
cudaEventRecord(stop);

所有这些操作与主机都是异步的，它们都被绑定到默认的流中。最后的 cudaMemcpyAsync 函数一旦被发布，
控制权将立即返回到主机。一旦控制权返回给主机，主机就可以做任何有用的计算，
而不必再依赖内核输出。在下面的代码段中，主机只是简单迭代，等待所有的 CUDA 操作完成时计数器加1。
在每次迭代中，主机线程查询停止事件。一旦事件完成，主机线程继续：

unsigned long int counter = 0;
while(cudaEventQuery(stop) == cudaErrorNotReady){
    counter++；
}
*/

int main(int argc, char *argv[])
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps,devID);
    printf("> %s running on", argv[0]);
    printf(" CUDA device [%s]\n", deviceProps.name);

    int num = 1 << 24;
    int nbytes = num * sizeof(int);
    float value = 10.0f;

    // allocate host memory
    float *h_a = 0;
    cudaMallocHost((void **)&h_a,nbytes);
    memset(h_a,0,nbytes);

    // allocate device memory
    float *d_a = 0;
    cudaMalloc((void **)&d_a,nbytes);
    cudaMemset(d_a,255,nbytes);

    // set kernel launch configuration
    dim3 block = dim3(512);
    dim3 grid = dim3((num + block.x - 1) / block.x);

    // create cuda event handles
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    // asynchronously issue work to the GPU (all to stream 0)
    cudaMemcpyAsync(d_a,h_a,nbytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block>>>(d_a,value);
    cudaMemcpyAsync(h_a,d_a,nbytes,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while(cudaEventQuery(stop) == cudaErrorNotReady){
        counter++;
    }

    // print the cpu and gpu times
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
           counter);

    // check the output for correctness
    bool bFinalResults = (bool) checkResult(h_a, num, value);

    // release resources
    cudaEventDestroy(stop);
    cudaFreeHost(h_a);
    cudaFree(d_a);

    CHECK(cudaDeviceReset());

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

/*
$./asyncAPI 
> ./asyncAPI running on CUDA device [Tesla V100-SXM2-16GB]
CPU executed 22300 iterations while waiting for GPU to finish

在等待 GPU 操作完成时，主机线程执行了 22300 此迭代。

性能侧写如下：
$nvprof ./asyncAPI 
==76227== NVPROF is profiling process 76227, command: ./asyncAPI
> ./asyncAPI running on CUDA device [Tesla V100-SXM2-16GB]
CPU executed 11170 iterations while waiting for GPU to finish
==76227== Profiling application: ./asyncAPI
==76227== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.09%  8.5980ms         1  8.5980ms  8.5980ms  8.5980ms  [CUDA memcpy DtoH]
                   43.26%  6.7510ms         1  6.7510ms  6.7510ms  6.7510ms  [CUDA memcpy HtoD]
                    1.13%  176.26us         1  176.26us  176.26us  176.26us  kernel(float*, float)
                    0.52%  80.863us         1  80.863us  80.863us  80.863us  [CUDA memset]
      API calls:   50.62%  417.87ms         1  417.87ms  417.87ms  417.87ms  cudaDeviceReset
                   45.26%  373.67ms         1  373.67ms  373.67ms  373.67ms  cudaMallocHost
                    1.56%  12.912ms         1  12.912ms  12.912ms  12.912ms  cudaFreeHost
                    0.84%  6.9200ms     11171     619ns     587ns  8.3530us  cudaEventQuery
                    0.83%  6.8561ms         1  6.8561ms  6.8561ms  6.8561ms  cudaLaunchKernel
                    0.57%  4.7202ms       912  5.1750us     123ns  777.44us  cuDeviceGetAttribute
                    0.12%  1.0055ms         8  125.69us  1.5880us  988.53us  cuDeviceGetPCIBusId
                    0.05%  451.61us         1  451.61us  451.61us  451.61us  cudaMalloc
                    0.05%  444.67us         1  444.67us  444.67us  444.67us  cudaGetDeviceProperties
                    0.05%  430.60us         1  430.60us  430.60us  430.60us  cudaFree
                    0.01%  120.78us         1  120.78us  120.78us  120.78us  cudaMemset
                    0.01%  69.689us         8  8.7110us  5.1210us  17.323us  cuDeviceGetName
                    0.00%  33.802us         2  16.901us  11.064us  22.738us  cudaMemcpyAsync
                    0.00%  20.981us         1  20.981us  20.981us  20.981us  cudaEventCreate
                    0.00%  8.9880us         1  8.9880us  8.9880us  8.9880us  cudaEventDestroy
                    0.00%  5.5350us         1  5.5350us  5.5350us  5.5350us  cudaEventRecord
                    0.00%  4.5080us        16     281ns     190ns     475ns  cuDeviceGet
                    0.00%  4.0590us         8     507ns     311ns     868ns  cuDeviceTotalMem
                    0.00%  3.1780us         8     397ns     295ns     710ns  cuDeviceGetUuid
                    0.00%  1.5910us         3     530ns     329ns     803ns  cuDeviceGetCount
                    0.00%     476ns         1     476ns     476ns     476ns  cuModuleGetLoadingMode
*/

