/**
4.1 内存访问模式
大多数device 端数据访问都是从 global memory 开始的，并且多数 GPU 应用程序容易受memory bandwidth的限制。
因此，最大限度地利用global memory bandwidth 是调控kernel 函数性能的根本。
如果不能正确地调控全局内存的使用， 其他优化方案很可能也收效甚微。

为了在读写数据时达到最佳性能，内存访问操作必须满足一定的条件。 
CUDA 执行模型的显著特征之一就是指令必须以 wrap 为单位进行发布和执行
存储操作也是同样。在执行内存指令时，
wrap 中的每个thread 都提供了一个正在load或store的内存地址。
在wrap的32个线程中，
每个thread 都提出了一个包含请求地址的单一 memory access 请求，它并由一个或多个device memory transaction提供服务。
根据wrap中memory address的分布，memory access 可以被分成不同的模式。
本节中，将学习不同的内存访问模式，并学习如何实现最佳的global memory access
*/

/**
4.3.1 对齐与合并访问
如图4-6所示，global memory 通过 cache 来实现 load/store。
global memory 是一个逻辑内存空间，可以通过kernel 函数进行访问
所有的应用程序的数据最初存储在 DRAM 上，即物理设备内存中。
kernel 函数的memory access 请求
通常是在 DRAM 设备和片上内存间以128字节或32字节内存transaction来实现的

所有对global memory的访问都会通过L2 Cache，也有许多访问会通过 L1 Cache，
这取决于访问类型和 GPU 架构。
如果这两级 cache 都被用到，那么内存访问是由一个128字节的内存事务实现的。
如果只使用了L2 Cache，那么这个memory access 是由一个32字节的内存事务来实现的。
对global memory缓存其架构，如果允许使用一级缓存，那么可以在编译时选择启用或禁用 L1 Cache。

一行 L1 Cache 是 128个字节，它映射到设备内存中一个128字节的对齐段。
如果warp中的每个thread 请求一个4字节的值，那么每次请求就会获取128字节的数据，
这恰好与 cache 行和 device memory 段的大小相契合。
因此在优化应用程序时，需要注意device memory access 的两个特性：
1. 对齐内存访问
2. 合并内存访问

------------------------------------             -------------------------------------
|SM0                                |            |SM1                                |
| |-------------------------------| |            | |-------------------------------| |
| |       寄存器                   | |            | |       寄存器                   | |
| |--------------------------------| |           | |--------------------------------| |
|  ^       ^          ^        ^     |           |  ^       ^          ^        ^     |
|  |       |          |        |     |           |  |       |          |        |     |
|  v       v          v        v     |           |  v       v          v        v     |
| |-----| |--------| |-----|  |----| |           | |-----| |--------| |-----|  |----| |
| |SMEM | |L1 Cache| |只读  |  |常量| |           | |SMEM | |L1 Cache| |只读  |  |常量| |
| ------- ---------- -------   ----- |           | ------- ---------- -------   ----- |
|-------------------------------------|          |-------------------------------------|
                    ^                                                  ^
                    |                                                  |
                    v                                                  v
|-----------------------------------------------------------------------------------------|
|二级缓存(L2 Cache)                                                                        |
|-----------------------------------------------------------------------------------------|                   
                    ^                                                  ^
                    |                                                  |
                    v                                                  v
|-----------------------------------------------------------------------------------------|
|DRAM                                                                                     |
|-----------------------------------------------------------------------------------------|

当device memory transaction的第一个地址是用于transaction服务的缓存粒度的偶数倍时（如32字节的二级缓存
或128字节的一级缓存时），就会出现对齐内存访问。运行非对齐的加载会造成带宽浪费。

当一个wrap中全部的32个thread 访问一个连续的内存块时，就会出现合并内存访问。

对齐合并内存访问的理想状态是wrap从对齐内存地址开始访问一个连续的内存块。
为了最大化global memory throughput，组织内存操作进行对齐合并是很重要的。

图4-7描述了对齐与合并内存的load 操作。在这种情况下，只需要一个128字节的memory transaction从device memory 中读取事务。
图4-8展示了非对齐和未合并的memory access。在这种情况下，可能需要3个128字节
的memory transaction来从device memory 中 load data：
一个在offset 为 0的地方开始，读取连续地址之后的数据；
一个在offset 为 256的地方开始，读取连续地址之前的数据；
另一个在offset 为128的地方开始，读取大量的数据。
注意在memory transaction 之前和之后获取的大部分字节将不能被使用，这样会造成带宽浪费

对齐+连续：
内存地址：
                 128             160             192             224             256
         | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
         ------------------------------------------------------------------------------------
                  | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 
                  v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v 
线程ID:           0 1 2 3 4 5 6 7 8 9 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3
                                      0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1

非对齐+非连续：
对齐+连续：
内存地址：
                 128             160             192             224             256
         | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
         ----------------------------------------------------------------_________/--------
           \        / | | | | | | | | | | | | | | | | | | | | | | | | | / | | |        / 
           |        v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v      /
线程ID:         > 0 1 2 3 4 5 6 7 8 9 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3  <
                                      0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1

一般来说，需要优化内存transaction的效率：用最少的transaction次数满足最多的内存请求。
transaction数量和throughput的需求随设备的计算能力变化。
*/

/*
4.3.2 global memory load（全局内存读取）
在 SM 中，数据通过以下3种缓存/缓冲路径进行传输，
具体使用何种方式取决于引用了哪种类型的device memory：
1. 一级和二级缓存
2. 常量缓存
3. 只读缓存
L1/L2 Cache 是默认路径。想要通过其他两种路径传递数据需要应用程序显式地说明，
但要想提升性能还要取决于使用的访问模式。global memory load 操作是否会通过 L1 Cache 取决于两个因素：
1. 设备的计算能力
2. 编译器选项

在 Fermi GPU (计算能力为2.x) 和 Kepler K40及以后得GPU(计算能力在3.5及以上)中，
可以通过编译器标志启用或禁用global memory load 的 L1 Cache。
默认情况下，在 Fermi 设备上对于global memory load 加载可以用 L1 Cache，在 K40 及以上
GPU 中禁用。以下标志通知编译器禁用 L1 Cache:
-Xptxas -dlcm=cg

如果 L1 Cache 被禁用，所有对 global memory 的 load 请求将直接进入到 L2 Cache
如果 L2 Cache 缺失，则由 DRAM 完成请求。每一次memory transaction可由一个、两个或四个部分执行，
每个部分有 32 个字节。一级缓存也可以使用下列标识符直接启用：
-Xptxas -dlcm=ca

设置这个标志后，global memory load 请求首先尝试通过L1 Cache。
如果 L1 Cache 缺失，该请求转向 L2 Cache。如果 L2 Cache 缺失，则请求由 DRAM 完成
在这种模式下，一个memory load 请求由一个128字节的设备内存事务实现。

在 Kepler K10/K20 和 K20X GPU 中，L1 Cache 不用来缓存全局内存加载。
L1 Cache专门用于缓存register溢出到load memory中的数据。

内存加载访问模式：
1. 内存加载可以分为两类：
（1）缓存加载（启用 L1 Cache）
（2）没有缓存的加载（禁用 L1 Cache)
2. 内存加载的访问模式有如下的特点：
（1）有缓存与没有缓存：如果启用 L1 Cache，则内存加载被缓存
（2）对齐与非对齐：如果内存访问的第一个地址是32字节的倍数，则对齐加载
（3）合并与非合并：如果wrap 访问一个连续的数据块，则加载合并
*/

/*
4.3.2.1 缓存加载
缓存加载操作经过 L1 Cache，
在粒度为128字节的 L1 Cache 行上由设备内存事务进行传输。
缓存加载可以分为对齐/非对齐以及合并/非合并

1. 理想情况：对齐与合并内存访问。wrap中所有thread请求的地址都在128字节的缓存行的范围内。
完成 memory load 操作只需要一个128字节的事务。总的使用率为100%
在这个书屋中没有未使用的数据。
wrap 中的地址：                           | | | | | | | | | | | | | | | | | | | | | |
                                         v v v v v v v v v v v v v v v v v v v v v v
0--------32--------64--------96--------128--------160--------192--------224--------256--------288--------320

2. 另一种情况：访问是对齐的，引用的地址不是连续的线程 ID，而是128 字节范围内的随机值。由于wrap 中线程请求的地址
仍然在一个缓存行范围内，所以只需要一个128字节的书屋来完成这一加载操作。总线利用率仍然是100%
并且只有当每个线程请求在128字节范围内有4个不同的字节时，这个事务中才没有未使用的数据。
wrap 中的地址：                           | | | \ / | | | | | | | | | | | | | | \ / |
                                         v v v v > v v v v v v v v v v v v v v v > v
0--------32--------64--------96--------128--------160--------192--------224--------256--------288--------320

3. 另一种情况：wrap请求32个连续4个字节的非对齐数据元素。在global memory 中
wrap 的线程请求的地址落在两个128字节范围内。
因为当启用L1 Cache时，
由 SM 执行的物理加载内存操作继续在128个字节的界线上对齐，
所以要求有两个128字节的事务来执行这段内存加载操作。
总线利用率为50%，并且在这两个事务中加载的字节由一半是未使用的。
wrap 中的地址：                           / / / / / / / / | | | | | | | | | | | | | |
                    _____________________________________v v v v v v v v v v v v v v v v
                    v  v  v   V  v V v V
0--------32--------64--------96--------128--------160--------192--------224--------256--------288--------320

4. 另一种情况：wrap 中所有线程都请求相同的地址。因为被引用的字节落在一个缓存行范围内，
所以只需要请求一个内存事务。但总线利用率非常低。如果加载的值是4字节的，则总线利用率是4字节请求/128字节加载=3.125%

wrap 中的地址：                           \ \ \ \ \ \ \ \ \ \ / / / / / / / / / / / /
                                                            |
                                                            v 
0--------32--------64--------96--------128--------160--------192--------224--------256--------288--------320

5. 最坏的情况：线程中线程束请求分散于全局内存中的32个4字节地址。尽管线程请求的字节总数仅为128个字节，
但地址要占用 N 个缓存行（0<N<=32）。完成一次内存加载操作需要申请 N 次内存事务。

CPU L1 Cahce 与 GPU L1 Cahce 的差异：
CPU 的 L1 Cache 优化了时间和空间局部性。GPU 的 L1 Cache是专门为空间局部性而不是为时间局部性设计的。
频繁访问一个 L1 Cache中的内存位置也不会增加数据留在缓存中的概率。
*/

/*
4.3.2.2 没有缓存的加载
没有缓存的加载不会经过 L1 Cache，它在内存段的粒度上（32个字节）而非缓存池的粒度（128个字节）执行。
这是更细粒度的加载，可以为非对齐或非合并的内存访问带来更好的总线利用率。
1. 理想情况：对齐与合并内存访问。128个字节请求的地址占用了4个内存segment，总线利用率为100%

2. 另一种情况：内存访问时对齐的且内存访问不连续，而是在128个字节的范围内随机进行。
只要每个线程请求唯一的地址，那么地址将占用4个内存段，并且不会有加载浪费。
这样的随机访问不会抑制内核性能。

3. 另一种情况：wrap 请求32个连续的4字节元素但加载没有对齐到128个字节的边界。
请求的地址最多落在5个内存段内，总线利用率至少为80%。
与这些类型的请求缓存加载相比，使用非缓存加载会提升性能，这是因为加载了更少的未请求字节。

4.另一种情况：线程束中所有线程请求相同的数据。
地址落在一个内存段内，总线的利用率为请求的4字节/加载的32字节=12.5%，在这种情况下，非缓存加载性能也是优于缓存加载性能的。

5.最坏的情况：wrap 请求32个分散在global memory 中的4字节字。由于请求的128个字节最多落在N个32字节的内存分段内，而不是 N个128字节的缓存行内，
所以相比于缓存加载，即便是最坏的情况也有所改善。
*/



/*
4.3.3.3 非对齐读取的实例
因为访问模式往往是由应用程序实现的一个算法来决定的，
所以对于某些应用程序来说，合并内存加载是一个挑战。
然而，在大多数情况下，使用某些方法可以帮助对齐应用程序的内存访问。

为了说明kernel 函数中非对齐访问对性能的影响，使用向量加法代码为例。
去掉内存加载操作，指定一个偏移量。注意在下面的kernel 中使用了两种索引。
新的索引 k 由给定的偏移量上移，由于偏移量的值可能会导致加载出现非对齐加载。
只有加载数组 A 和数组 B 的操作会用到索引 k。
对数组 C 的写操作仍使用原来的索引 i，以确保写入访问保持对齐。
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

//offset 默认为0
__global__ void readOffset(float *A, float *B, float *C,const int n,int offset){
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k<n){
        C[i] = A[k] + B[k];
    }
}

//为了保证修改后kernel 函数的正确性，主机代码也要做出相应的修改：
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void initialData(float *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 100.0f;
    }

    return;
}


void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

/*
4.3.2.4 只读缓存
只读缓存最初是预留给texture 内存加载使用的。对于计算能力为3.5 及以上的 GPU 来说，
只读缓存也支持使用 global memory load，代替 L1 Cache

只读缓存的加载粒度是32个字节。通常，对分散读取来说，这些更细粒度的加载要优于 L1 Cache

有两种方式可以指导内存通过只读缓存进行读取：
使用函数：__ldg
在间接引用的指针上使用修饰符

例如，考虑下面的拷贝 kernel 函数：
*/

__global__ void copyKernel(int *out,int *in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}

/*
可以使用内部函数 __ldg 来通过只读缓存直接对数组进行读取访问：
*/
__global__ void copyKernel_ldg(int *out,const int *in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = __ldg(&in[idx]);
}

/*
也可以将常量 __restrict__修饰符应用到指针上。这些修饰符帮助nvcc编译器识别无别名指针（即专门用来访问特定数组的指针）
nvcc 将自动通过只读缓存指导无别名指针的加载。
*/
__global__ void copyKernel_restrict(int * __restrict__ out, const int * __restrict__ in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}

int main(int argc,char **argv){
    //setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting reduction at ",argv[0]);
    printf("device %d: %s ",dev,deviceProp.name);
    cudaSetDevice(dev);

    //set up array size
    int nElem = 1<<20; // total number of elements to reduce
    printf(" with array size %d\n",nElem);
    size_t nBytes = nElem * sizeof(float);

    //setu up offset for summary
    int blocksize = 512;
    int offset = 0;
    if(argc>1){
        offset = atoi(argv[1]);
    }

    if(argc>2){
        blocksize = atoi(argv[2]);
    }

    //execution configuration
    dim3 block (blocksize,1);
    dim3 grid ((nElem+block.x-1)/block.x,1);

    //allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    //initialize host array
    initialData(h_A,nElem);
    memcpy(h_B,h_A,nBytes);

    //summary at host side
    sumArraysOnHost(h_A,h_B,hostRef,nElem,offset);

    //allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A,nBytes);
    cudaMalloc((float **)&d_B,nBytes);
    cudaMalloc((float **)&d_C,nBytes);

    //copy data from host to device
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);

    //  kernel 1:
    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup     <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
            block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
            block.x, offset, iElaps);
    CHECK(cudaGetLastError());
 
    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);
 
    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
 
    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;

}

/*
通过以全局加载效率为指标，可以验证非对齐访问为性能损失的原因：

全局加载效率 = （请求的全局内存加载吞吐量）/ (所需的全局内存加载吞吐量)

所需的全局内存加载吞吐量包括重新执行的内存加载指令，这个指令不只需要一个内存事务
而请求的全局内存加载吞吐量并不需要如此。

可以使用nvprof 获取 gld_efficiency 指标，其中nvprof 带有readSegment 测试用例和不同偏移量值：
sudo nvprof --devices 0 --metrics gld_transactions,gld_efficiency ./readSegment 0
==24035== NVPROF is profiling process 24035, command: ./readSegment 0
./readSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
==24035== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(float*, float*, float*, int, int)" (done)
warmup     <<< 2048,  512 >>> offset    0 elapsed 0.029500 sec
Replaying kernel "readOffset(float*, float*, float*, int, int)" (done)
readOffset <<< 2048,  512 >>> offset    0 elapsed 0.010605 sec
==24035== Profiling application: ./readSegment 0
==24035== Profiling result:
==24035== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: readOffset(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262144      262144      262144
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
    Kernel: warmup(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262144      262144      262144
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

sudo nvprof --devices 0 --metrics gld_transactions,gld_efficiency ./readSegment 32
==21713== NVPROF is profiling process 21713, command: ./readSegment 32
./readSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
==21713== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(float*, float*, float*, int, int)" (done)
warmup     <<< 2048,  512 >>> offset   32 elapsed 0.032184 sec
Replaying kernel "readOffset(float*, float*, float*, int, int)" (done)
readOffset <<< 2048,  512 >>> offset   32 elapsed 0.012449 sec
==21713== Profiling application: ./readSegment 32
==21713== Profiling result:
==21713== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: readOffset(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262136      262136      262136
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
    Kernel: warmup(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262136      262136      262136
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

$sudo nvprof --devices 0 --metrics gld_transactions,gld_efficiency ./readSegment 128
==25996== NVPROF is profiling process 25996, command: ./readSegment 128
./readSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
==25996== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(float*, float*, float*, int, int)" (done)
warmup     <<< 2048,  512 >>> offset  128 elapsed 0.029170 sec
Replaying kernel "readOffset(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "readOffset(float*, float*, float*, int, int)" (done)
==25996== Profiling application: ./readSegment 1280.010449 sec
==25996== Profiling result:
==25996== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: readOffset(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262112      262112      262112
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
    Kernel: warmup(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262112      262112      262112
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

$sudo nvprof --devices 0 --metrics gld_transactions,gld_efficiency ./readSegment 16
==27911== NVPROF is profiling process 27911, command: ./readSegment 16
./readSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
==27911== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(float*, float*, float*, int, int)" (done)
warmup     <<< 2048,  512 >>> offset   16 elapsed 0.031512 sec
Replaying kernel "readOffset(float*, float*, float*, int, int)" (done)
readOffset <<< 2048,  512 >>> offset   16 elapsed 0.012117 sec
==27911== Profiling application: ./readSegment 16
==27911== Profiling result:
==27911== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: readOffset(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262140      262140      262140
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
    Kernel: warmup(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      262140      262140      262140
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

$sudo nvprof --devices 0 --metrics gld_transactions,gld_efficiency ./readSegment 11 （发生错位，非对齐）
==28344== NVPROF is profiling process 28344, command: ./readSegment 11
./readSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
==28344== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(float*, float*, float*, int, int)" (done)
warmup     <<< 2048,  512 >>> offset   11 elapsed 0.029004 sec
Replaying kernel "readOffset(float*, float*, float*, int, int)" (done)
readOffset <<< 2048,  512 >>> offset   11 elapsed 0.009843 sec
==28344== Profiling application: ./readSegment 11
==28344== Profiling result:
==28344== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: readOffset(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      327676      327676      327676
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%
    Kernel: warmup(float*, float*, float*, int, int)
          1                          gld_transactions                  Global Load Transactions      327676      327676      327676
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%

非对齐访问，造成了内存事务量增加了25%（262144-》327676）



          */

/**
没有缓存的加载的整体性能略低于缓存访问的整体性能。
L1 Cache miss对于非对齐访问的性能影响更大。
如果启用 L1 Cache，一个非对齐访问可能将数据存到一级缓存，这个L1 Cache用于后续的非对齐内存访问
但是，如果没有 L1 Cache，那么每次非对齐请求需要多个内存事务，并且对将来的请求没有作用

使用没有缓存的整体加载时间并没有减少，但是全局加载效率提高了
确实是这样，但这种结果只针对这种测试实例
随着谁被占用率的提高，没有缓存的加载可帮助提高总线的整体利用率
对于没有缓存的非对齐加载模式来说，未使用的数据传输量可能会显著减少。
*/



