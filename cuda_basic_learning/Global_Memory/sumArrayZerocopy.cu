/*
4.2.4 Zero Copy 内存

通常来说，主机不能直接访问设备变量，同时设备也不能直接访问主机变量
但是有一个例外，即零拷贝内存。主机和设备都可以访问零拷贝内存。

GPU 线程可以直接访问零拷贝内存。在 CUDA 核函数中使用零拷贝内存有以下几个优势：

1.当设备内存不足时，可以利用主机内存
2. 避免主机和设备间的显式的数据传输
3. 提高 PCIe 传输率

当使用零拷贝内存来共享主机和设备间的数据时，
必须同步主机和设备间的内存访问，同时更改主机和设备的零拷贝内存中的数据将导致不可预知的后果

零拷贝内存是固定（不可分页）内存，该内存映射到设备地址空间中。可以通过下列函数创建一个到固定内存的映射：
cudaError_t cudaHostAlloc(void **pHost,size_t count, unsigned int flags);
这个函数分配了 count 字节的主机内存，该内存是页面锁定的且设备可访问的。
用这个函数分配的内存必须用cudaFreeHost 函数来释放。flags 参数可以对已分配内存的特殊属性进行进一步的配置，具体如下：

cudaHostAllocDefault: 使cudaHostAlloc 函数的行为与 cudaMallocHost 保持一致。
cudaHostAllocPortable: 可以返回能被所有 CUDA context 使用的固定内存，而不仅仅是执行内存分配的那一个
cudaHostAllocWriteCombined：返回了写结合的内存，该内存可以在某些系统配置上通过 PCIe总线更快地进行传输，但是它在大多数主机上不能被有效地读取
因此，写结合内存对缓冲区来说是一个很好的选择，该内存通过设备使用映射的固定内存或主机到设备的传输。
cudaHostAllocMapped：零拷贝内存的最明显的标志。该标志返回，可以实现主机写入和设备读取映射到设备地址空间中的主机内存。
可以使用下列函数获取映射到固定内存的设备指针：
cudaError_t cudaHostGetDevicePointer(void **pDevice,void *pHost,unsigned int flags);
该函数返回了一个在pDevice中的设备指针，该指针可以在设备上被引用以访问映射得到的固定主机内存。
如果设备不支持映射得到的固定内存，该函数将失效。flag 将留作以后使用，现在它必须被设为0。

在进行频繁的读写操作时，使用零拷贝内存作为设备内存的补充将显著降低性能。因为，每一次映射到主机内存的传输必须经过 PCIe 总线。
与device global memory 相比，延迟也显著增加。

下面利用矩阵求和来验证这一性能波动。
为了测试零拷贝内存读操作的性能，可以给数组 A 和 B分配零拷贝内存，并在设备内存上为数组 C 分配内存

主函数包含了两部分：第一部分为从设备内存加载数据及存储数据到设备内存；第二部分为从零拷贝内存加载数据，并将数据存储到设备内存中。
首先需要检查设备是否支持固定内存映射。
为了允许 kernel 函数从零拷贝内存中读取数据，需要将数组 A 和 B 分配作为映射的固定内存。
然后，可以直接在host 上初始化 数组 A 和 B。不需要将他们传输给设备内存。
接下来，获取供kernel 函数使用映射的固定内存的设备指针。一旦内存被分配并初始化，就可以调用 kernel 函数了

*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates the use of zero-copy memory to remove the need to
 * explicitly issue a memcpy operation between the host and device. By mapping
 * host, page-locked memory into the device's address space, the address can
 * directly reference a host array and transfer its contents over the PCIe bus.
 *
 * This example compares performing a vector addition with and without zero-copy
 * memory.
 */

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                    gpuRef[i], i);
            break;
        }
    }

    return;
}

void initialData(float *ip, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrays(float *A,float *B, float *C, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc,char **argv){
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // get device properties
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // check if support mapped memory
    if(!deviceProp.canMapHostMemory){
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("Using Device %d: %s ", dev, deviceProp.name);

    // set up data size of vectors
    int ipower = 10;

    if (argc > 1) ipower = atoi(argv[1]);

    int nElem = 1 << ipower;

    size_t nBytes = nElem * sizeof(float);

    if (ipower < 18)
    {
        printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower,
               (float)nBytes / (1024.0f));
    }
    else
    {
        printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower,
               (float)nBytes / (1024.0f * 1024.0f));
    }

    // part 1: using device memory
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((float **)&d_A,nBytes));
    CHECK(cudaMalloc((float **)&d_B,nBytes));
    CHECK(cudaMalloc((float **)&d_C,nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice));

    // set up execution configuration
    int iLen = 512;

    dim3 block (iLen);
    dim3 grid ((nElem+block.x - 1)/block.x);

    sumArrays<<<grid,block>>>(d_A,d_B,d_C,nElem);
    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    // free host memory
    free(h_A);
    free(h_B);

    // part 2: using zerocopy memory for array A and B
    // allocate zerocpy memory
    CHECK(cudaHostAlloc((void **)&h_A,nBytes,cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&h_B,nBytes,cudaHostAllocMapped));
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    // pass the pointer to device
    CHECK(cudaHostGetDevicePointer((void **)&d_A,(void *)h_A,0));
    CHECK(cudaHostGetDevicePointer((void **)&d_B,(void *)h_B,0));

    // add at host side for result checks
    sumArraysOnHost(h_A,h_B,hostRef,nElem);

    // execute kernel with zero copy memory
    sumArraysZeroCopy<<<grid,block>>>(d_A,d_B,d_C,nElem);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free  memory
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);

    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;

}

/*
$nvprof ./sumZerocopy (1024个元素)
==24365== NVPROF is profiling process 24365, command: ./sumZerocopy
Using Device 0: Tesla V100-SXM2-16GB Vector size 1024 power 10  nbytes    4 KB
==24365== Profiling application: ./sumZerocopy
==24365== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.00%  7.8720us         1  7.8720us  7.8720us  7.8720us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.76%  4.4800us         2  2.2400us  2.2400us  2.2400us  [CUDA memcpy DtoH]
                   19.67%  3.8720us         2  1.9360us  1.8880us  1.9840us  [CUDA memcpy HtoD]
                   17.56%  3.4560us         1  3.4560us  3.4560us  3.4560us  sumArrays(float*, float*, float*, int)
      API calls:   87.09%  327.91ms         3  109.30ms  3.6260us  327.91ms  cudaMalloc

$nvprof ./sumZerocopy 22 （4M 个元素）
==30262== NVPROF is profiling process 30262, command: ./sumZerocopy 22
Using Device 0: Tesla V100-SXM2-16GB Vector size 4194304 power 22  nbytes   16 MB
==30262== Profiling application: ./sumZerocopy 22
==30262== Profiling result:
Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.52%  5.9200ms         2  2.9600ms  2.8938ms  3.0262ms  [CUDA memcpy HtoD]
                   34.83%  4.8486ms         2  2.4243ms  2.4066ms  2.4420ms  [CUDA memcpy DtoH]
                   22.16%  3.0850ms         1  3.0850ms  3.0850ms  3.0850ms  sumArraysZeroCopy(float*, float*, float*, int)
                    0.49%  68.447us         1  68.447us  68.447us  68.447us  sumArrays(float*, float*, float*, int)


比较 sumArraysZeroCopy kernel 函数 和 sumArrays kernel 函数的运行时间。
当处理 1024 个元素时，从零拷贝内存读取的kernel 函数运行时间比只使用设备内存的 kernel 函数慢了2.28 倍
不过需要注意，从设备到主机传输数据的时间（DtoH）也应该计算在这两个 kernel 函数的运行时间里面。
因为它们都使用了 cudaMemcpy 来更新主机端数据，计算结果的操作在设备上执行。
同时我们发现，当计算 4M 个元素时，因为数据量的增大，从零拷贝内存读取的kernel 函数运行时间比只使用设备内存的 kernel 函数慢了 86.78 倍

从结果中可以看出，如果向共享 host 和 device 端少量的数据，零拷贝内存可能会是一个不错的选择
因为它简化了编程并且具有较好的性能。
对于由 PCIe 总线连接的离散 GPU 上的更大数据集来说，零拷贝内存不是一个好的选择，它会导致性能的显著下降
*/

/*
零拷贝内存

有两种常见的异构计算系统架构：集成架构和离散架构
在集成架构中，CPU 和 GPU 集成在一个芯片上，并且在物理地址上共享主存。在这种架构中，由于不需要再 PCIe 总线上备份，
所以零拷贝内存在性能和可编程性方面可能更优。

对于通过 PCIe 总线将设备连接到主机的离散系统而言，零拷贝内存只在特殊情况下才有优势

因为映射的固定内存在主机和设备之间是共享的，你必须同步内存访问来避免任何潜在的数据冲突
这种数据冲突一般由多线程异步访问相同的内存而引起的。

注意不要过度使用零拷贝内存。由于其延迟较高，从零拷贝内存中读取的device kernel 函数可能很慢。
*/


