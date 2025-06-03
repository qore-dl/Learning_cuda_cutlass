#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100000
#define NSTREAM 4

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

/*
6.5 流回调
流回调是另一种可以到 CUDA 流中排列等待的操作。一旦流回调之前的流操作全部完成，
被流回调指定的主机端函数就会被 CUDA runtime 所调用。此函数由应用程序提供，
并允许任意主机端逻辑插入到 CUDA 流中。流回调是一种 CPU 和 GPU 同步机制。
回调功能十分强大，因为它们是第一个 GPU 操作的例子，此操作用于在主机系统上
创建工作，这与在本书中阐述这一点的 CUDA 概念完全相反。

流回调函数是由应用程序提供的一个主机函数，并在流中使用以下的 API 函数注册：
cudaError_t cudaStreamAddCallback(cudaStream_t stream,cudaStreamCallback_t callback,
                                void *userData,unsigned int flags);

此函数为提供的流添加了一个回调函数。在流中所有先前排队的操作完成后，回调函数才能在主机上执行。
每使用 cudaStreamAddCallback 一次，只执行一次回调，并阻塞队列中排在其后面的工作，
直到回调函数完成。当它被 CUDA runtime 调用时，回调函数会通过调用它的流，
并且会有错误代码来表明是否有 CUDA 错误的发生。还可以使用 cudaStreamAddCallback 的
userData 参数，指定传递给回调函数的应用程序函数。flags 参数在后面将会使用，
目前没有任何意义；因此，必须将它设置为 0。在所有流中先前的工作都完成后，排在空流中的回调队列才会被执行。

对于回调函数有两个限制：
1. 从回调函数中不可以调用 CUDA 的 API 函数
2. 在回调函数中不可以执行同步

一般来说，对互相关联或与其他 CUDA 操作相关的回调顺序做任何假设都是有风险的，可能导致代码不稳定。

下面的代码示例中，在 4 个流都执行 4 个内核后，为每个流的末尾添加回调函数：
my_callback。只在当每个流中的所有工作都完成后，回调函数才开始在主机上运行：

void CUDART_CB my_callback(cudaStream_t stream,cudaError_t status,void *data){
    printf("callback from stream %d\n",*((int *)data));
}

为每个流添加回调的代码如下：
for(int i=0;i<n_streams;i++){
    stream_ids[i] = i;
    kernel_1<<<grid,block,0,streams[i]>>>();
    kernel_2<<<grid,block,0,streams[i]>>>();
    kernel_3<<<grid,block,0,streams[i]>>>();
    kernel_4<<<grid,block,0,streams[i]>>>();
    cudaStreamAddCallback(streams[i],my_callback,(void *)(stream_ids+i),0);
}
*/

void CUDART_CB my_callback(cudaStream_t stream,cudaError_t status,void *data)
{
    printf("callback from stream %d\n", *((int *)data));
}

int main(int argc, char **argv)
{
    int n_streams = NSTREAM;

    if (argc > 1) n_streams = atoi(argv[1]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting...\n", argv[0]);
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
    setenv(iname, "8", 1);
    char *ivalue = getenv(iname);
    printf ("> %s = %s\n", iname, ivalue);
    printf ("> with streams = %d\n", n_streams);


    // Allocate and initialize an array of stream handles
    
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(
                                cudaStream_t));

    for(int i = 0; i< n_streams; i++){
        cudaStreamCreate(&(streams[i]));
    }

    dim3 block (1);
    dim3 grid  (1);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    int stream_ids[n_streams];
    
    cudaEventRecord(start_event,0);

    for(int i=0;i < n_streams; i++)
    {
        stream_ids[i] = i;
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
        cudaStreamAddCallback(streams[i],my_callback,(void *)(stream_ids+i),0);
    }

    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start_event,stop_event);
    printf("Measured time for parallel execution = %.3fs\n",
           elapsed_time / 1000.0f);

    // release all stream
    for(int i=0;i<n_streams;i++){
        cudaStreamDestroy(streams[i]);
    }

    free(streams);

    /*
     * cudaDeviceReset must be called before exiting in order for profiling and
     * tracing tools such as Nsight and Visual Profiler to show complete traces.
     */
    CHECK(cudaDeviceReset());

    return 0;
}

/*
$./callback 
> ./callback Starting...
> Using Device 0: Tesla V100-SXM2-16GB
> Compute Capability 7.0 hardware with 80 multi-processors
> CUDA_DEVICE_MAX_CONNECTIONS = 8
> with streams = 4
callback from stream 0
callback from stream 1
callback from stream 2
callback from stream 3
Measured time for parallel execution = 0.003s
*/


