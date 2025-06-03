#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 300000
#define NSTREAM 4


/*
6.2 并行内核执行
前面已经解释了流、事件和同步的概念以及 API，接下来用几个例子来演示一下。
第一个示例演示了如何使用多个流并发运行多个核函数。
这个简单的例子将介绍并发执行的几个基本问题，包括以下几个方面：
1. 使用深度优先或广度优先方法的调度工作
2. 调整硬件工作队列
3. 在 Kepler 设备和 Fermi 设备上避免虚假的依赖关系
4. 检测默认流的阻塞行为
5. 在非默认流之间添加依赖关系
6. 检查资源使用使用是如何影响并发的
*/

/*
6.2.1 非空流中的并发内核
在本节中，将使用 NVIDIA 的可视化性能分析器（nvvp）可视化并发核函数执行。
在该例子中使用的kernel函数包括在设备上仿真有用工作的虚拟计算。
这确保了内核驻留在 GPU 中时间足够长，以使重叠在可视化性能分析器中更加明显。
注意这个例子使用了多个相同的 kernel 函数（被称为kernel_1,kernel_2,...):
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

    // char* iname = "CUDA_DEVICE_MAX_CONNECTIONS"; //设置设备与主机硬件连接的上限，即设置硬件工作队列的上限
    // setenv(iname,"32",1);
    // char *ivalue = getenv(iname);
    // printf("%s=%s\n",iname,ivalue);

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
        kernel_1<<<grid, block, 0, streams[0]>>>();
        kernel_2<<<grid, block, 0, streams[0]>>>();
        kernel_3<<<grid, block, 0, streams[0]>>>();
        kernel_4<<<grid, block, 0, streams[0]>>>();
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