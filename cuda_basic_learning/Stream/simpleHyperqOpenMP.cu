#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 300000
#define NSTREAM 4

/*
6.2.3 使用 OpenMP 的调度操作
前面的示例中，使用单一的主机线程将异步 CUDA 操作调度到多个流中。
本节的示例将使用多个主机线程将操作调度到多个流中，并使用一个线程来管理一个流

OpenMP 是 CPU 的并行编程模型，它使用编译器指令来识别并行区域。
支持 OpenMP 指令的编译器可以将用作如何并行化应用程序的提示。
用很少的代码，在主机上就可以实现多核并行。

在使用 OpenMP 的同时使用 CUDA，不仅可以提高便携性和生产效率，
而且还可以提高主机代码的性能。在simpleHyperQ 例子中，我们使用了
一个循环调度操作，与此不同，我们使用了OpenMP 线程调度操作到不同的流中，
具体方法如下所示：
omp_set_num_threads(n_streams);
#pragma omp parallel
{
 int i = omp_get_thread_num();
 kernel_1<<<grid,block,0,streams[i]>>>();
 kernel_2<<<grid,block,0,streams[i]>>>();
 kernel_3<<<grid,block,0,streams[i]>>>();
 kernel_4<<<grid,block,0,streams[i]>>>();
}

OpenMp 函数 omp_set_num_threads 用来指定在 OpenMP 并行区域里要用到的 CPU 核心数量。
编译器指令 #pragma omp parallel 将花括号之间的代码标记为并行部分。
omp_get_thread_num 函数为每个主机线程返回一个唯一的线程 ID，将该 ID 用作streams
数组中的索引，用来创建 OpenMP 线程和 CUDA 流间的一对一映射。
编译时使用 -Xcompiler 选项将标识传递给支持 OpenMP 的主机编译器：
nvcc -O3 -Xcomplier -fopenmp simpleHyperqOpenmp.cu -o simpleHyperqOpenmp -lgomp

什么时候从 OpenMP中调度并行 CUDA 操作时有用的？在一般情况下，如果每个流在内核执行之前、期间或之后有额外的工作待完成，
那么它可以在包含同一个 OpenMP 并行区域中，并且跨流和线程进行重叠。
这样做更明显地说明了每个 OpenMP 线程中的主机工作与同一个线程中启动的流 CUDA 操作是相关的，
并且可以为了优化性能简化代码的书写。
*/


__global__ void kernel_1()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char **argv)
{

    int n_streams= NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin

    char *iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname,"32",1);
    char *ivalue = getenv(iname);
    printf ("%s = %s\n", iname, ivalue);

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name,
           n_streams);
    cudaSetDevice(dev);

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

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
    
    for(int i=0;i<n_streams;i++){
        cudaStreamCreate(&(streams[i]));
    }


    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration    
    dim3 block(iblock);
    dim3 grid(isize/iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record start event
    cudaEventRecord(start,0);

    // dispatch job with depth first ordering using OpenMP

    /*
    在使用 OpenMP 的同时使用 CUDA，不仅可以提高便携性和生产效率，
    而且还可以提高主机代码的性能。在simpleHyperQ 例子中，我们使用了
    一个循环调度操作，与此不同，我们使用了OpenMP 线程调度操作到不同的流中，
    具体方法如下所示：
    */

    omp_set_num_threads(n_streams);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }



    // record stop event
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time,start,stop);
    printf("Measured time for parallel execution = %.fs\n",
           elapsed_time / 1000.0f);

    // release all stream

    for(int i=0;i<n_streams;i++){
        cudaStreamDestroy(streams[i]);
    }

    free(streams);

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}