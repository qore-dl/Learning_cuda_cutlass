/*
4.2.3 固定内存

分配的host memory 默认是 pageable (可分页)，
它的意思也就是因页面错误导致的操作，
该操作按照操作系统的要求将主机虚拟内存上的数据移动到不同的物理位置。
虚拟内存给人一种比实际可用内存大得多的假象，就如同L1 缓存好像比实际可用的 on-chip memory 大得多一样。

GPU 不能在可分页主机内存上安全地访问数据，因为当主机操作系统在物理位置上移动该数据时，它无法控制。
当从可分页主机内存传输到设备内存时，CUDA 驱动程序首先分配临时页面锁定的或固定的主机内存，
将主机源数据复制到固定内存中，然后从固定内存传输数据给设备内存，如下图所示：

分页数据传输：

Device
                     DRAM
                       ^
-----------------------|------------------
Host                   |
                       |
                       |
可分页内存 ---------> 固定内存


CUDA runtime 允许使用以下的指令直接分配固定主机内存：

cudaError_t cudaMallocHost(void **devPtr,size_t count);
这个函数分配了 count 字节的主机内存，这些内存是页面锁定的并且对设备来说是可访问的
由于固定内存能被设备直接访问，所以它能用比可分页内存高得多的带宽进行读写
然而，分配过多的固定内存可能会降低主机系统的性能，因为它减少了用于存储虚拟内存数据的可分页内存的数量
其中分页内存对主机系统是可用的，如下图所示：

固定数据传输：
Device
                     DRAM
                       ^
-----------------------|------------------
Host                   |
                       |
                       |
                    固定内存

下面的代码段用来分配固定主机内存，其中含错误检查和基本错误处理：

cudaError_t status = cudaMallocHost((void **)&h_aPinned,bytes);
if(status != cudaSuccess){
    fprintf(stderr,"Error returned from pinned host memory allocation\n");
    exit(1);
}

固定主机内存必须通过下述指令来释放：
cudaError_t cudaFreeHost(void *ptr);
可以试着在文件 memTransfer.cu 中用固定主机内存替换可分页内存。
*/

#include <cuda_runtime.h>
#include <stdio.h>

#include "common.h"

/*
 * An example of using CUDA's memory copy API to transfer data to and from the
 * device. In this case, cudaMalloc is used to allocate memory on the GPU and
 * cudaMemcpy is used to transfer the contents of host memory to an array
 * allocated using cudaMalloc. Host memory is allocated using cudaMallocHost to
 * create a page-locked host array.
 */

int main(int argc, char **argv){
    //set up device;
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    //memory size;
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));

    if(!deviceProp.canMapHostMemory){
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("%s starting at ",argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB canMap %d\n", dev,
           deviceProp.name, isize, nbytes / (1024.0f * 1024.0f),
           deviceProp.canMapHostMemory);
    
    
    
    // allocate pinned host memory
    float *h_a;
    CHECK(cudaMallocHost((float **)&h_a,nbytes));
    // malloc(nbytes);

    //allocate the device memory
    float *d_a;
    CHECK(cudaMalloc((float **)&d_a,nbytes));

    // initialize host memory
    memset(h_a,0,nbytes);

    for(unsigned int i=0;i<isize;i++){
        h_a[i] = 100.10f;
    }

    //transfer data from the host to the deivce
    CHECK(cudaMemcpy(d_a,h_a,nbytes,cudaMemcpyHostToDevice));

    //transfer data from the device to the host
    CHECK(cudaMemcpy(h_a,d_a,nbytes,cudaMemcpyDeviceToHost));
    
    //free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFreeHost(h_a));

    //reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/*
下面的输出清除地说明了使用固定内存与使用memTransfer生成的输出相比性能有了显著的提升。

$sudo nvprof ./pinMemTransfer
==99999== NVPROF is profiling process 99999, command: ./pinMemTransfer
./pinMemTransfer starting at device 0: Tesla V100-SXM2-16GB memory size 4194304 nbyte 16.00MB canMap 1
==99999== Profiling application: ./pinMemTransfer
==99999== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.21%  1.5707ms         1  1.5707ms  1.5707ms  1.5707ms  [CUDA memcpy HtoD]
                   47.79%  1.4379ms         1  1.4379ms  1.4379ms  1.4379ms  [CUDA memcpy DtoH]

sudo nvprof ./memTransfer
==123143== NVPROF is profiling process 123143, command: ./memTransfer
./memTransfer starting at device 0: Tesla V100-SXM2-16GB memory size 4194304 nbytes 16.00MB
==123143== Profiling application: ./memTransfer
==123143== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.25%  3.2156ms         1  3.2156ms  3.2156ms  3.2156ms  [CUDA memcpy DtoH]
                   44.75%  2.6044ms         1  2.6044ms  2.6044ms  2.6044ms  [CUDA memcpy HtoD]

使用可分页主机内存时传输最初耗时总计为5.82 ms，使用了pinmemory 后，耗时总计下降到3.0086 ms

主机与设备间的内存传输

与可分页内存相比，固定内存的分配和释放的成本更高，但是它可以为大规模数据传输提供了更高的传输吞吐量

相对于可分页的内存，使用固定内存获取的加速取决于设备的计算能力。例如，当传输超过 10MB 的数据时，
在 Fermi 设备上使用固定内存通常是更好的选择
许多小的传输批处理合并为一个更大的传输能提升性能。因为它减少了单位传输消耗。
主机和设备之间的数据传输有时候可以与 kernel 执行重叠，关于这个话题将在后文学习
应该尽可能地减少或重叠主机和设备之间的数据传输。

*/