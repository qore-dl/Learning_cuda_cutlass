#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 300000
#define NSTREAM 16


/*
6.2.4 用环境变量调整流行为
支持 Hyper-Q 的 GPU 在主机和每个 GPU 之间维护硬件工作队列，
消除虚假的依赖关系。Kepler 设备支持的硬件工作队列的最大数量是32。
然而，默认情况下并发硬件连接的数量被限制为8。减少了不需要全部32个工作队列的
应用程序的资源消耗。
可以使用 CUDA_DEVICE_MAX_CONNECTIONS 环境变量来调整并行硬件连接的数量。
对于 Kepler 设备而言，其上限是 32。
有几种设置改环境变量的方法。在Linux 中，可以根据 shell 的版本，
通过以下代码进行设置，对于 Bash 和 Bourne Shell,其代码如下：

export CUDA_DEVICE_MAX_CONNECTIONS=32

对于 C-shell，其代码如下：
setenv CUDA_DEVICE_MAX_CONNECTIONS 32
这个环境变量也可以直接在 C 主机程序中进行设定：
setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);

每个 CUDA 流都会被映射到单一的 CUDA 设备连接中。
如果流的数量超过了硬件连接的数量，多个流会共享一个硬件连接。
当多个流共享相同的硬件工作队列时，可能会产生虚假的依赖关系。

在支持 Hyper-Q 技术但没有足够硬件连接的平台上，要检查 CUDA 流的行为，
需要调整创建的流的数量：例如
#define NSTREAM 8 (使用 8 个 CUDA 流)
*/





__global__ void kernel_1(){
    double sum = 0.0;
    for(int i=0;i<N;i++){
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
这样做视为了在nvvp中更容易将不同内核的执行进行可视化。
*/

int main(int argc,char **argv){
    int n_streams = NSTREAM;
    int iblock = 1;
    int isize = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    // char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    // setenv (iname, "32", 1);
    // char *ivalue =  getenv (iname);
    // printf ("%s = %s\n", iname, ivalue);

    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS"; //设置设备与主机硬件连接的上限，即设置硬件工作队列的上限
    setenv(iname,"4",1); //将 CUA 设备硬件连接的数量设置为 4
    /*
    硬件连接数量上限小于使用的流的数量，
    因此，导致了分配在同一工作队列中的两个流之间出现了虚假的依赖关系。
    在一个硬件工作队列中，只要资源足够且前后op没有依赖关系（如在两个流上），则这个两个op（两个流）可以并发
    因此硬件工作队列上不要求串行，串行是由软件层面如流、事件等导致的。
    */
    char *ivalue = getenv(iname);
    printf("%s=%s\n",iname,ivalue);

    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name,
           n_streams);
    cudaSetDevice(dev);

    //check if device support hyper-q
    if ((deviceProp.major<3)||(deviceProp.major==3&&deviceProp.minor<5)){
        if(deviceProp.concurrentKernels == 0){
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }else{
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }
    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    
    /*
    首先必须要创建一组非空流。在这组非空流中，发布每个流中的内核启动应该 GPU 上同时运行，
    但是应不存在由于硬件资源限制而导致的虚假依赖关系。
    */
    //Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams*sizeof(cudaStream_t));
    for(int i=0;i<n_streams;i++){
        cudaStreamCreate(&streams[i]);
    }

    /*
    使用一个循环遍历所有的流，这样内核在每个流中都可以被调度：
    */
    // run kernel with more threads
    /*
    在实际的应用中，内核启动时通常会创建多个线程。
    通常，会创建数百到数千个线程。
    有了这么多线程，可用的硬件资源可能会成为并发的主要限制因素，
    因为它们阻止启动符合条件的内核。
    为了在活动中观察到这个行为，可以在该函数中改变执行配置，
    在每个块中使用多个线程，在每个网格中使用更多的块
    */
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
    /*
    在本例子中，为了计算运行时间，也创建了两个事件
    */
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record start event
    /*
    在启动所有的内核循环前，启动事件就已经被记录在默认流中了。
    而在所有的内核启动后，停止事件也被记录在默认流中。
    */
    cudaEventRecord(start,0);
    /*
    使用一个遍历循环所有的流，这样内核在每个流中都可以被调度
    */
    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        /*
        6.2.5 GPU资源的并发限制
        有限的内核资源可以抑制应用程序中可能出现的内核并发的数量。在之前的例子中，
        启动内核时只有一个线程，以避免并发时任何的硬件限制。
        因此，每个内核只需要少量的设备计算资源：
        kernel_1<<<1,1,0,streams[i]>>>();

        在实际的应用中，内核启动时通常会创建多个线程。
        通常，会创建数百到数千个线程。
        有了这么多线程，可用的硬件资源可能会成为并发的主要限制因素，
        因为它们阻止启动符合条件的内核。
        为了在活动中观察到这个行为，可以在该函数中改变执行配置，
        在每个块中使用多个线程，在每个网格中使用更多的块
        */
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }
    /*
    这些内核启动的执行配置被指定为单一线程块中的单一线程，
    以确保有足够的 GPU 资源能并发运行所有的内核。
    因为每个内核启动相对于主机来说都是异步的，
    所以可以通过使用单一主机线程同时调度多个内核到不同的流中
    */
    
    /*
    在同步停止事件后，可以计算运行时间：
    */
    // record stop event
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    //  // calculate elapsed time
    // CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    // printf("Measured time for parallel execution = %.3fs\n",
    //        elapsed_time / 1000.0f);
    // calculate elapsed time
    cudaEventElapsedTime(&elapsed_time,start,stop);
    printf("Measured time for parallel execution = %fs\n",
           elapsed_time / 1000.0f);
    
    // release all stream
    for(int i=0;i<n_streams;i++){
        // cudaStreamDe
        // cudaStreamDestroy(streams[i])
        cudaStreamDestroy(streams[i]);
    }

    free(streams);
    // // destroy events
    // CHECK(cudaEventDestroy(start));
    // CHECK(cudaEventDestroy(stop));

    //destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}