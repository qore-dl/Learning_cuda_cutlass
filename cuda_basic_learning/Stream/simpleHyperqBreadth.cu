#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 300000
#define NSTREAM 16

/*
6.2.2 Fermi GPU 上的虚假依赖关系
当 GPU 不支持 Hyper-Q 时，内核最终会限制并发一起运行。

例如，在 Fermi GPU 上，四个流不能同时启动，这是由于共享硬件工作队列造成的。
为什么 流 i+1 能够在流 i 开始其最后任务是开始它的第一个任务呢？
因为任务是在两个不同的流中，所以它们之间没有依赖关系。
当流 i 的最后一个任务被启动时，CUDA runtime 从工作队列中调度下一个任务，
这是 流 i+1 的第一个任务。因为每个流的第一个任务不依赖于之前的任何任务，
并且有可用的 SM，所以它可以立即启动。之后，调度流 i+1 的第二个任务，
然而它对第一个任务的依赖却阻止它被执行，这就会导致任务执行再次被阻塞。

这种虚假的依赖关系是由主机调度内核顺序引起的。
该应用程序使用深度优先的方法，在下一个流启动前，在该流中启动全系列的操作。
利用深度优先方法得到的工作队列中的任务顺序如图6-7所示。由于所有流被多路复用
到一个硬件工作队列中，所以前面的流就连续阻塞了后面的流：

主机调度任务的顺序：深度优先方法
-----------------------------------------------------------------------------------------------》
K1_S1|K2_S1|K3_S1|K4_S1|K1_S2|K2_S2|K3_S2|K4_S2|K1_S3|K2_S3|K3_S3|K4_S3|K1_S4|K2_S4|K3_S4|K4_S4|
                            图 6-7

在 Fermi GPU 上，为了避免虚假的依赖关系，可以用广度优先的方法从主机中调度工作：
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
    //设置设备与主机硬件连接的上限，即设置硬件工作队列的上限
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname,"4",1);
    char *ivalue = getenv(iname);
    printf("%s=%s\n",iname,ivalue);
   /*
    硬件连接数量上限小于使用的流的数量，
    因此，导致了分配在同一工作队列中的两个流之间出现了虚假的依赖关系。
    在一个硬件工作队列中，只要资源足够且前后op没有依赖关系（如在两个流上），则这个两个op（两个流）可以并发
    因此硬件工作队列上不要求串行可以并行，串行是由软件层面如流、事件等导致的。
    */

    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name,n_streams);

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
    
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams*sizeof(cudaStream_t));

    for(int i=0;i<n_streams;i++){
        cudaStreamCreate(&streams[i]);
    }

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
    dim3 block (iblock);
    dim3 grid  (isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record start event
    cudaEventRecord(start,0);

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

    for(int i=0;i<n_streams;i++){
        kernel_1<<<grid,block,0,streams[i]>>>();
    }

    for(int i=0;i<n_streams;i++){
        kernel_2<<<grid,block,0,streams[i]>>>();
    }

    for(int i=0;i<n_streams;i++){
        kernel_3<<<grid,block,0,streams[i]>>>();
    }

    for(int i=0;i<n_streams;i++){
        kernel_4<<<grid,block,0,streams[i]>>>();
    }
    /*
    因此，在硬件连接工作队列数量少于软件流的情况下，使用广度优先的方法发布内核，
    可以去除虚假的依赖关系。
    */

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time,start,stop);

    printf("Measured time for parallel execution = %fs\n",
           elapsed_time / 1000.0f);
    
    // release all stream
    for(int i=0;i<n_streams;i++){
        cudaStreamDestroy(streams[i]);
    }

    free(streams);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
    return 0;
}
/*
每个kernel 中一个线程：
$./simpleHyperqBreadth 
CUDA_DEVICE_MAX_CONNECTIONS=4
> Using Device 0: Tesla V100-SXM2-16GB with num_streams=8
> Compute Capability 7.0 hardware with 80 multi-processors
> grid 1 block 1
Measured time for parallel execution = 0.000425s

每个kernel 中有512个线程：
$./simpleHyperqBreadth 32 1
CUDA_DEVICE_MAX_CONNECTIONS=4
> Using Device 0: Tesla V100-SXM2-16GB with num_streams=32
> Compute Capability 7.0 hardware with 80 multi-processors
> grid 8 block 512
Measured time for parallel execution = 0.000717s
*/