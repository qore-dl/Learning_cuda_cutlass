/*
4.2.4 统一虚拟寻址
计算能力为 2.0 及以上版本的设备支持一种特殊的寻址方式，被称为统一虚拟寻址（UVA）。

UVA，在 CUDA 4.0 被引入，支持 64 位 Linux 系统。有了 UVA，主机内存和设备内存可以共享同一个虚拟地址空间
如下图所示：

---------  -----------  -----------
|CPU 内存| |CPU 0 内存|  |CPU 1 内存|
---------  -----------  -----------
  ^          ^            ^
  |          |            |
  v          v            v
  GPU       GPU 0        GPU 1
  |           |           | PCIe 总线
  |-----------|-----------|--------
            无 UVA：多内存空间

--------------------------------
|CPU 内存   CPU0 内存   CPU 1 内存|
----------------------------------
  ^          ^            ^
  |          |            |
  v          v            v
  GPU       GPU 0        GPU 1
  |           |           | PCIe 总线
  |-----------|-----------|--------
            UVA：单内存空间

在 UVA 之前，你需要管理哪些指针指向主机内存和哪些指针指向设备内存。
有了 UVA，由指针指向的内存空间对应用程序代码来说是透明的。

通过 UVA，由 cudaHostAlloc 分配的固定主机内存具有相同的主机和设备指针
因此，可以将返回的指针直接传递给 kernel 函数
无 UVA 时需要以下几个操作：

1. 分配映射的固定主机内存
2. 使用 CUDA runtime 函数获取映射到固定内存的设备指针
3. 将设备指针传递给 kernel 函数


有了 UVA，无须获取设备指针或者管理物理上完全相同的两个指针。UVA 会进一步简化sumArrayZeroCpy.cu中的操作。
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
    // allocate zerocpy memory at the host side
    CHECK(cudaHostAlloc((void **)&h_A,nBytes,cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&h_B,nBytes,cudaHostAllocMapped));
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    /*
    使用UVA 后无须以下的操作：
     // // pass the pointer to device
    // CHECK(cudaHostGetDevicePointer((void **)&d_A,(void *)h_A,0));
    // CHECK(cudaHostGetDevicePointer((void **)&d_B,(void *)h_B,0));
    即无须：2. 使用 CUDA runtime 函数获取映射到固定内存的设备指针
    3. 将设备指针传递给 kernel 函数
    这是因为，有了 UVA，无须获取设备指针或者管理物理上完全相同的两个指针。
    UVA 会进一步简化sumArrayZeroCpy.cu中的操作。
    */

   
    // add at host side for result checks
    sumArraysOnHost(h_A,h_B,hostRef,nElem);

    // execute kernel with zero copy memory
    //直接：invoke the kernel with zero-cpy memory (此时主机指针与设备指针实现统一寻址，二者完全一致)
    sumArraysZeroCopy<<<grid,block>>>(h_A,h_B,d_C,nElem);
    //注意，从 cudaHostAlloc 函数返回的指针被直接传递给 kernel 函数


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


$sudo nvprof ./sumArrayZerocpyUVA
==11393== NVPROF is profiling process 11393, command: ./sumArrayZerocpyUVA
Using Device 0: Tesla V100-SXM2-16GB Vector size 1024 power 10  nbytes    4 KB
==11393== Profiling application: ./sumArrayZerocpyUVA
==11393== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.06%  7.9360us         1  7.9360us  7.9360us  7.9360us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.78%  4.5120us         2  2.2560us  2.2400us  2.2720us  [CUDA memcpy DtoH]
                   19.71%  3.9040us         2  1.9520us  1.8880us  2.0160us  [CUDA memcpy HtoD]
                   17.45%  3.4560us         1  3.4560us  3.4560us  3.4560us  sumArrays(float*, float*, float*, int)

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

$sudo nvprof ./sumArrayZerocpyUVA 22
==11606== NVPROF is profiling process 11606, command: ./sumArrayZerocpyUVA 22
Using Device 0: Tesla V100-SXM2-16GB Vector size 4194304 power 22  nbytes   16 MB
==11606== Profiling application: ./sumArrayZerocpyUVA 22
==11606== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.14%  6.0002ms         2  3.0001ms  2.9328ms  3.0673ms  [CUDA memcpy HtoD]
                   35.45%  5.0470ms         2  2.5235ms  2.5062ms  2.5408ms  [CUDA memcpy DtoH]
                   21.92%  3.1212ms         1  3.1212ms  3.1212ms  3.1212ms  sumArraysZeroCopy(float*, float*, float*, int)
                    0.48%  68.736us         1  68.736us  68.736us  68.736us  sumArrays(float*, float*, float*, int)

我们发现，与之前有给设备指定映射固定内存的指针的 zero-copy 的版本相比，
本次 利用 UVA 特性使得指向固定内存的主机指针与设备指针一致的版本具有相似的性能结果
使用更少的代码取得了相同结果，这提高了应用程序的可读性和可维护性

*/

/*
4.2.6 统一内存寻址

在 CUDA 6.0 中，引入了 "统一内存寻址" 这一新特性，它用于简化 CUDA 编程模型中的内存管理

统一内存中创建了一个托管内存池，内存池中已分配的空间可以用相同的内存地址（即指针）在 CPU 与 GPU 上进行访问。
底层系统在统一内存空间中自动在主机和设备之间进行数据传输。
这种数据传输对于应用程序而言是透明的，这大大简化了程序代码

统一内存寻址依赖于 UVA 的支持，但它们是完全不同的技术。
UVA位系统中的所有处理器提供了一个单一的虚拟内存地址空间
但是 UVA 不会将数据从一个物理位置迁移到另一个物理位置，这是统一内存寻址的一个特有功能。

统一内存寻址提供了一个"单指针到数据" 模型，在概念上它类似于零拷贝内存。
但是零拷贝内存在主机内存中进行分配，因此由于受到 PCIe 总线上访问零拷贝内存的影响，
kernel 函数将具有较高的延迟。另一方面，统一内存寻址将内存和执行空间分离，
因此可以根据需要将数据透明地传输到主机或设备上，用以提升局部性和性能。

托管内存指的是由底层系统自动分配的统一内存，与特定于设备的分配内存可以互操作。
如它们的创建都使用cudaMalloc API
因此，你可以在kernel 函数中使用两种类型的内存：由系统控制的托管内存，以及由应用程序明确分配和调用的未托管内存
所有在设备内存上有效的 CUDA 操作同样也适用于 托管内存。其主要区别是主机也能够引用和访问托管内存。

托管内存可以被静态地分配也可以实现动态分配。
可以通过添加 __managed__ 注释，静态声明一个设备变量作为托管变量。
但这个操作智能在文件范围和全局范围内进行。该变量可以从主机或设备代码中直接被引用：

__device__ __managed__ int y;

还可以使用下述的 CUDA runtime 函数动态分配托管内存：
cudaError_t cudaMallocManaged(void **devPtr,size_t size,unsigned int flags=0);

这个函数分配 size 字节的托管内存，并用 devPtr 返回一个指针。该指针在所有设备和主机上都是有效的。
使用托管内存的程序行为与使用未托管内存的程序副本行为在功能上是一致的。
但是使用托管内存的程序可以利用自动数据传输和重复指针消除功能。

在 CUDA 6.0 中，设备代码不能调用 cudaMallocManaged 函数，
所有的托管内存必须在主机端动态声明或者在全局范围内静态声明。
在 4.5 节中，将会详细说明 CUDA 的统一内存寻址机制。

*/




