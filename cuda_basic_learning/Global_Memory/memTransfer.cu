/*
4.2 内存管理

CUDA 编程的内存管理与 C 语言的类似
需要程序员显式地管理主机和设备之间的数据移动
随着 CUDA 版本的升级，NVIDIA 正在系统地实现主机和设备内存空间的统一
但对于大多数应用程序来说，仍需要手动移动数据
这一领域中的最新进展将在本章的4.2.6中进行介绍
现在，工作重点在于如何使用 CUDA 函数来显示地管理内存和数据移动：

1. 分配和释放设备内存
2. 在主机和设备之间传输数据

为了达到最优性能，CUDA 提供了在主机端准备设备内存的函数，并且显式地向设备传输数据和从设备中获取数据
*/

/*
4.2.1 内存分配和释放
CUDA 编程模型假设了一个包含一个主机和一个设备的异构系统。
每一个异构系统有自己独立的内存空间。
kernel 函数在设备内存空间中运行
CUDA runtime 提供函数以分配和释放设备内存
可以在主机上使用下列函数分配global memory：
cudaError_t cudaMalloc(void **devPtr,size_t count);

这个函数在设备上分配了 count 字节的 global memory，并使用 devptr 指针返回该内存空间的地址
所分配的内存支持任何变量类型，包括整型、浮点数类型、布尔类型等。
如果cudaMalloc函数执行失败则返回cudaErrorMemoryAllocation。
在已分配的global memory 中的值不会被清除。
你需要用从 host 上传输的数据来填充所分配的global memory
或使用下列函数将其初始化：
cudaError_t cudaMemset(void *devPtr,int value,size_t count);

这个函数用存储在变量value中的值来填充从设备 global memory 地址 devPtr处开始的 count 字节的数据。
一旦一个应用程序不再使用已分配的global memory，那么可以使用以下代码释放该内存空间：
cudaError_t cudaFree(void *devPtr);

这个函数释放了devPtr指向的global memory，该内存必须在此前使用了一个设备分配函数（如 cudaMalloc）来进行分配。
否则，它将返回一个错误 cudaErrorInvalidDevicePointer。如果该地址空间已经被释放，那么cudaFree也返回一个错误。

设备内存的分配和释放操作成本较高，所以应用程序应重利用设备内存，以减少对整体性能的影响。
*/

/*
4.2.2 内存传输
一旦分配好了global memory，可以使用下列函数从主机向设备传输数据：
cudaError_t cudaMemcpy(void *dst,const void *src,size_t count,enum cudaMemcpyKind kind);

这个函数从内存位置 src 复制了 count 字节到内存位置 dst。变量kind 指定了复制的方向，可以有下列取值：
cudaMemcpyHostToHost;
cudaMemcpyHostToDevice;
cudaMemcpyDeviceToHost;
cudaMemcpyDeviceToDevice;

如果指针dst 和 src 与 kind 指定的复制方向不一致，那么 cudaMemcpy 的行为就是未定义行为。这个函数在大多数情况下都是同步的。

下列函数是一个使用 cudaMemcpy 的例子。这个例子展示了在主机和设备之间来回地传输数据。
使用 cudaMalloc 分配 global memory，使用 cudaMemcpy 将数据传输到设备
传输方向由 cudaMemcpyHostToDevice 指定。
然后使用 cudaMemcpy 将数据传回主机，方向由 cudaMemcpyDeviceToHost 指定：
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc,char **argv){
    //set up device
    int dev = 0;
    cudaSetDevice(dev);

    //memory size;
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting at ",argv[0]);
    printf("device %d: %s memory size %d nbytes %5.2fMB\n",dev,deviceProp.name,
            isize,nbytes/(1024.0f*1024.0f));
    
    //allocate the host memory
    float *h_a = (float *)malloc(nbytes);

    //allocate the device memory
    float *d_a;
    cudaMalloc((float **)&d_a,nbytes);

    //initialize the host memory
    for(unsigned int i=0;i<isize;i++){
        h_a[i] = 0.5f;
    }

    //transfer data from the host to the deivce
    cudaMemcpy(d_a,h_a,nbytes,cudaMemcpyHostToDevice);

    //transfer data from the device to the host
    cudaMemcpy(h_a,d_a,nbytes,cudaMemcpyDeviceToHost);
    
    //free memory
    cudaFree(d_a);
    free(h_a);

    //reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}

/*
sudo nvprof ./memTransfer
==123143== NVPROF is profiling process 123143, command: ./memTransfer
./memTransfer starting at device 0: Tesla V100-SXM2-16GB memory size 4194304 nbytes 16.00MB
==123143== Profiling application: ./memTransfer
==123143== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.25%  3.2156ms         1  3.2156ms  3.2156ms  3.2156ms  [CUDA memcpy DtoH]
                   44.75%  2.6044ms         1  2.6044ms  2.6044ms  2.6044ms  [CUDA memcpy HtoD]
      API calls:   86.69%  351.41ms         1  351.41ms  351.41ms  351.41ms  cudaMalloc
                   10.84%  43.935ms         1  43.935ms  43.935ms  43.935ms  cudaDeviceReset
                    1.57%  6.3756ms         2  3.1878ms  2.6950ms  3.6806ms  cudaMemcpy
                    0.72%  2.9028ms       808  3.5920us     140ns  179.95us  cuDeviceGetAttribute
                    0.09%  364.95us         1  364.95us  364.95us  364.95us  cudaGetDeviceProperties
                    0.07%  303.27us         1  303.27us  303.27us  303.27us  cudaFree
                    0.01%  29.863us         8  3.7320us  2.5590us  9.0270us  cuDeviceGetName
                    0.01%  23.243us         8  2.9050us  1.0760us  13.062us  cuDeviceGetPCIBusId
                    0.00%  4.2550us         1  4.2550us  4.2550us  4.2550us  cudaSetDevice
                    0.00%  3.5410us        16     221ns     139ns     804ns  cuDeviceGet
                    0.00%  2.0920us         8     261ns     181ns     563ns  cuDeviceTotalMem
                    0.00%  1.6120us         8     201ns     186ns     242ns  cuDeviceGetUuid
                    0.00%  1.2450us         3     415ns     230ns     783ns  cuDeviceGetCount
                    0.00%     393ns         1     393ns     393ns     393ns  cuModuleGetLoadingMode

HtoD:主机到设备
DtoH: 设备到主机

下图表示了 CPU 内存和 GPU 内存间的连接性能：

CPU<-------------->CPU端memory
^
|
|PCIe: 8GB/s
|
v
GPU<-------------->GPU端Memory
        GDDR5
        144GB/s

从图中可以看到 GPU 芯片和on-chip GDDR5 GPU memory之间的理论峰值带宽非常高
例如，在 Fermi C2050 GPU 上，这一理论值可以达到 144 GB/s
CPU 和 GPU 之间通过 PCIe Gen2总线相连。这种连接的理论带宽要低得多
例如为 8GB/s（PCIe Gen3 总线的最大理论限制值是 16 GB/s）
这种差距意味着如果管理不当的话，主机和设备间的数据传输会降低应用程序的整体性能

因此，CUDA 编程的一个基本原则应是尽可能地减少主机与设备之间的传输。
*/
