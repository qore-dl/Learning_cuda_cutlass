#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 

#define N 300000
#define NSTREAM 16

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
6.2.7 创建流间依赖关系
在理想情况下，流之间不应该有非计划之内的依赖关系（即虚假的依赖关系）。
然而，在复杂的应用程序中，引入流间的依赖关系是很有用的，
它可以在一个流中阻塞操作，直到另一个流中的操作完成。
事件可以用来添加流间依赖关系。
假如我们想让一个流中的工作在其他所有流中的工作都完成后才开始执行，
那么就可以使用事件来创建流之间的依赖关系。
首先，将标志设置为 cudaEventDisableTiming，创建同步事件，代码如下：
cudaEvent_t *kernelEvent = (cudaEvent_t *)malloc(n_streams*sizeof(cudaeEvent_t));
for(int i=0;i<n_streams;i++){
    cudaEventCreateWithFlags(&kernelEvent[i],cudaEventDisableTiming);
}
接下来，使用 cudaEventRecord 函数，在每个流完成时记录不同的事件。然后，
使用 cudaStreamWaitEvent 使得最后一个流（即streams[n_streams-1]）等待
其他所有流：
*/

int main(int argc,char **argv){
    int n_streams = NSTREAM;
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
    printf("> Using Device %d: %s with num_streams %d\n", dev, deviceProp.name,
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
    
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams*sizeof(cudaStream_t));


    for(int i=0;i < n_streams; i++){
        cudaStreamCreate(&(streams[i]));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block (iblock);
    dim3 grid (isize/iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // create event array:
    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *)malloc(n_streams * sizeof(cudaEvent_t));

    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaEventCreateWithFlags(&(kernelEvent[i]),
                    cudaEventDisableTiming));
    }

    // record start event
    cudaEventRecord(start,0);


    // dispatch job with depth first ordering
    for(int i=0;i<n_streams;i++){
        kernel_1<<<grid,block,0,streams[i]>>>();
        kernel_2<<<grid,block,0,streams[i]>>>();
        kernel_3<<<grid,block,0,streams[i]>>>();
        kernel_4<<<grid,block,0,streams[i]>>>();

        cudaEventRecord(kernelEvent[i],streams[i]);
        cudaStreamWaitEvent(streams[n_streams-1],kernelEvent[i],0);

    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.fs\n",
           elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0 ; i < n_streams ; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
        CHECK(cudaEventDestroy(kernelEvent[i]));
    }

    free(streams);
    free(kernelEvent);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}

/*
$./simpleHyperqDependence 32 1
CUDA_DEVICE_MAX_CONNECTIONS = 32
> Using Device 0: Tesla V100-SXM2-16GB with num_streams 32
> Compute Capability 7.0 hardware with 80 multi-processors
> grid 8 block 512
Measured time for parallel execution = 0.000817s

$./simpleHyperqDepth 32 1
CUDA_DEVICE_MAX_CONNECTIONS=32
> Using Device 0: Tesla V100-SXM2-16GB with num_streams=32
> Compute Capability 7.0 hardware with 80 multi-processors
> grid 8 block 512
Measured time for parallel execution = 0.000791s

注意，最后一个流，即streams[n_streams-1]，在其他所有流完成后才能开始启动工作。
*/

