#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example implements matrix element-wise addition on the host and GPU.
 * sumMatrixOnHost iterates over the rows and columns of each matrix, adding
 * elements from A and B together and storing the results in C. The current
 * offset in each matrix is stored using pointer arithmetic. sumMatrixOnGPU2D
 * implements the same logic, but using CUDA threads to process each matrix.
*/

 /*
    同步
    栅栏同步是一个原语，它在许多并行编程语言中都很常见。在 CUDA 中，同步可以在两个级别执行：
    系统级：等待host 和 device 完成所有的工作
    block级：在device执行过程中等待一个 block 中所有 block 到达同一点

    对于 host 来说，由于许多 CUDA API 调用和所有的kernel 启动不是同步的，
    cudaDeviceSynchronize 函数可以用来阻塞 host 应用程序，直到所有的 CUDA 操作 (复制、kernel 函数运行) 完成：
    cudaError_t cudaDeviceSynchronize(void);

    这个函数可能会从先前的异步 CUDA 操作中返回错误。
    因为在一个block中，thread warp 以一个未定义的顺序被执行，
    CUDA 提供了一个使用 block 局部栅栏来同步它们的执行的功能。使用以下函数在kernel中标记同步点：
    __device__ void __syncthreads(void);

    当 __syncthreads 被调用时，
    在同一个 block 中每个 thread 都必须等待直至该block中所有其他 thread 都已经达到这个同步点。
    在栅栏之前，所有线程产生的所有gloabl memory 和 shared memory access (访问)，
    将会在栅栏后对block中所有其他的线程可见。该函数可以协调同一个 block 中线程之间的通信，
    但它强制 warp 空闲，从而可能对性能产生负面影响。

    block 中的线程可以通过 shared memory 和 registers 来共享数据。当thread 之间共享数据时，
    要避免竞争条件。竞争条件或危险，是指多个线程无序地访问相同的内存位置。
    例如，当一个位置的无序读发生在写操作之后时，写后读竞争会发生。
    因为读和写之间没有顺序，所以读应该在写前还是写后的加载值是未定义的。
    其他竞争条件的例子由读后写或写后写。

    当block中的线程在逻辑上并行运行时，在物理上并不是所有的线程都可以在同一时间执行。
    如果线程 A 试图 读取由线程 B在不同线程束中写的数据，
    若使用了适当的同步，只需要确定线程 B 已经写完就可以了。
    否则，会出现竞争条件。

    在不同的block 之间的线程没有线程同步。
    block 间的同步：唯一安全的方法是在每个 kernel 执行结束端使用全局同步点，
    也就是说全局同步后，终止当前的 kernel 函数，开始执行新的 kernel 函数

    不同 block 中的thread 不允许相互同步，因此 GPU可以按照任意顺序执行block，这使得 CUDA 程序在大规模并行 GPU 上是可扩展的
*/

/*
可扩展性

对于任何并行应用程序而言，可扩展性是一个理想的特性。可扩展性意味着并行应用程序提供了额外的硬件资源。
相对于增加的资源，并行应用程序会产生加速。例如，若一个 CUDA 程序是在 两个 SM中是可扩展的，则与在一个 SM 中运行相比，
在 两个 SM 中运行会使得运行时间减半。一个可扩展的并行程序可以高效地使用所有的计算资源以提高性能。
可扩展性意味着增加的计算核心可以提高性能。
串行代码本身是不可扩展的，因为在成千上万的内核上运行一个串行单线程应用程序，对性能是没有影响的。并行代码有可扩展的潜能。
但真正的可扩展性取决于算法设计和硬件特性。

能够在可变数量的计算核心上执行相同的应用程序代码的能力被称为透明可扩展性。
一个透明的可扩展平台拓宽了现有应用程序的应用范围，并减少了开发人员的负担
因为它们可以避免新的或不同的硬件产生的变化。可扩展性比效率更重要。
一个可扩展但效率很低的系统可以通过简单添加硬件核心来处理更大的工作负载
一个效率很高但是不可扩展的系统很快会达到实现性能的上限。

CUDA 内核启动时，Block 分布在多个 SM 中。
grid 中的 block 以并行或连续或任意的顺序被执行。这种独立性使得 CUDA 程序在任意数量的计算核心间可以扩展。
*/

/*
并行性的表现实验
为更好地理解程序执行的本质，将使用不同的执行配置分析 sumMatrixOnGPU2D kernel 函数
使用 nvprof 配置指标，可以有助于理解为什么有些 grid/block的维数组合比其他的组合更好。
*/

void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            
            ic[ix] = ia[ix] + ib[ix];
        }
        ic += nx;
        ib += nx;
        ia += nx;
    }
    return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY){
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if(ix < NX && iy < NY){
        // 每次计算有3个内存操作：两个内存加载和一个内存存储。
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv){
    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    CHECK(cudaSetDevice(dev));

    //set up matrix size
    int nx = 1<<14;
    int ny = 1<<14;

    int nxy = nx*ny;
    int nBytes = nxy*sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);

    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A,nxy);
    initialData(h_B,nxy);
    double iElaps = seconds() - iStart;

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
    iElaps = seconds() - iStart;
     printf("sumMatrixOnHost elapsed %f s\n", iElaps);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA,nBytes));
    CHECK(cudaMalloc((void **)&d_MatB,nBytes));
    CHECK(cudaMalloc((void **)&d_MatC,nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB,h_B,nBytes,cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;

    if(argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block (dimx,dimy);
    dim3 grid ((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    // execute the kernel
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    sumMatrixOnGPU2D<<<grid,block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f s\n", grid.x,
           grid.y,
           block.x, block.y, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef,d_MatC,nBytes,cudaMemcpyDeviceToHost));

    //check device results
    checkResult(hostRef,gpuRef,nxy);

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

     // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

     
}

/*
1.
$./sumMatrix 32 32
sumMatrixOnHost elapsed 0.320475 s
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.004041 s

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.849480    0.849480    0.849480

2.
$./sumMatrix 32 16
sumMatrixOnHost elapsed 0.317903 s
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> elapsed 0.003921 s

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.878777    0.878777    0.878777

3.
$./sumMatrix 16 32
sumMatrixOnHost elapsed 0.315567 s
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> elapsed 0.004092 s

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.879418    0.879418    0.879418

4.
$./sumMatrix 16 16
sumMatrixOnHost elapsed 0.315000 s
sumMatrixOnGPU2D <<<(1024,1024), (16,16)>>> elapsed 0.003950 s

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.912735    0.912735    0.912735




第二种情况中的block 数量比 第一种情况的多，所以device 就可以有更多活跃的线程束。
其原因可能是第二种情况与第一种情况相比有更高的可实现占用率和更好的性能。

第四种情况有最高的可实现占用率，但它不是最快的。因此更高的占用率不一定意味着更高的性能。
肯定有其他因素限制 GPU 的性能。
*/

/*
用 nvprof 检测内存操作:
    // 每次计算有3个内存操作：两个内存加载和一个内存存储。-》 内存操作的瓶颈更大概率在
        C[idx] = A[idx] + B[idx];

用 gld_throught 指标检查kernel的内存读取效率，从而可以得到每个执行配置的差异：
1.
==60430== NVPROF is profiling process 60430, command: ./sumMatrix 32 32
sumMatrixOnHost elapsed 0.318364 s
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.016019 s
==60430== Profiling application: ./sumMatrix 32 32
==60430== Profiling result:
==60430== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  504.88GB/s  504.88GB/s  504.88GB/s


2. 
==63607== NVPROF is profiling process 63607, command: ./sumMatrix 32 16
sumMatrixOnHost elapsed 0.311272 s
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> elapsed 0.016073 s
==63607== Profiling application: ./sumMatrix 32 16
==63607== Profiling result:
==63607== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  520.66GB/s  520.66GB/s  520.66GB/s

3.
==69042== NVPROF is profiling process 69042, command: ./sumMatrix 16 32
sumMatrixOnHost elapsed 0.311878 s
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> elapsed 0.015969 s
==69042== Profiling application: ./sumMatrix 16 32
==69042== Profiling result:
==69042== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  495.90GB/s  495.90GB/s  495.90GB/s

4. 
==78613== NVPROF is profiling process 78613, command: ./sumMatrix 16 16
sumMatrixOnHost elapsed 0.310917 s
sumMatrixOnGPU2D <<<(1024,1024), (16,16)>>> elapsed 0.016071 s
==78613== Profiling application: ./sumMatrix 16 16
==78613== Profiling result:
==78613== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  515.42GB/s  515.42GB/s  515.42GB/s

第二种情况达到了最大的内存吞吐量，但是差距很小，更高的load 吞吐量不一定意味着更高的性能

接下来使用 gld_efficiency 指标检测全局加载效率：
gld_efficiency: 请求的全局内存负载吞吐量与所需的全局内存负载吞吐量的比率
即被请求的全局加载吞吐量占所需的全局加载吞吐量的比值。
它衡量了应用程序的加载操作利用设备内存带宽的程度，结果总结如下：

1.
==95788== NVPROF is profiling process 95788, command: ./sumMatrix 32 32
sumMatrixOnHost elapsed 0.309975 s
==95788== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.151988 s
==95788== Profiling application: ./sumMatrix 32 32
==95788== Profiling result:
==95788== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

2.
==101193== NVPROF is profiling process 101193, command: ./sumMatrix 32 16
sumMatrixOnHost elapsed 0.310398 s
==101193== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> elapsed 0.140950 s
==101193== Profiling application: ./sumMatrix 32 16
==101193== Profiling result:
==101193== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

3.
==102938== NVPROF is profiling process 102938, command: ./sumMatrix 16 32
sumMatrixOnHost elapsed 0.311200 s
==102938== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(1024,512), (16,32)>>> elapsed 0.144994 s
==102938== Profiling application: ./sumMatrix 16 32
==102938== Profiling result:
==102938== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

4.
==103694== NVPROF is profiling process 103694, command: ./sumMatrix 16 16
sumMatrixOnHost elapsed 0.310305 s
==103694== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(1024,1024), (16,16)>>> elapsed 0.148054 s
==103694== Profiling application: ./sumMatrix 16 16
==103694== Profiling result:
==103694== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

最后两种情况的共同特征：在最内层维数中block的大小是thread warp size 的一半
对grid 和 block 启发式算法来说，最内层的维数应该总是thread warp大小的倍数
后续将讨论半个线程束大小的block是如何影响性能的。
*/

/*
增大并行性
从之前的发现可以总结出，一个block的最内层维数（block.x）应该是线程束大小的倍数。
这样能极大地提高加载效率：
问题：
1.调整block.x 会进一步增加 load throughput吗
2. 有其他方法可以增大并行性吗

现在已经建立了性能基准，可以通过测试sumMatrix 使用更大范围的线程配置来回答这些问题：
$./sumMatrix 64 2
sumMatrixOnHost elapsed 0.305171 s
sumMatrixOnGPU2D <<<(256,8192), (64,2)>>> elapsed 0.003987 s

$./sumMatrix 64 4
sumMatrixOnHost elapsed 0.307733 s
sumMatrixOnGPU2D <<<(256,4096), (64,4)>>> elapsed 0.003966 s

$./sumMatrix 64 8
sumMatrixOnHost elapsed 0.309961 s
sumMatrixOnGPU2D <<<(256,2048), (64,8)>>> elapsed 0.003950 s

$./sumMatrix 128 2
sumMatrixOnHost elapsed 0.316288 s
sumMatrixOnGPU2D <<<(128,8192), (128,2)>>> elapsed 0.003944 s

$./sumMatrix 128 4
sumMatrixOnHost elapsed 0.315690 s
sumMatrixOnGPU2D <<<(128,4096), (128,4)>>> elapsed 0.003930 s

$./sumMatrix 128 8
sumMatrixOnHost elapsed 0.315341 s
sumMatrixOnGPU2D <<<(128,2048), (128,8)>>> elapsed 0.003947 s

$./sumMatrix 256 2
sumMatrixOnHost elapsed 0.316724 s
sumMatrixOnGPU2D <<<(64,8192), (256,2)>>> elapsed 0.003951 s

$./sumMatrix 256 4
sumMatrixOnHost elapsed 0.316812 s
sumMatrixOnGPU2D <<<(64,4096), (256,4)>>> elapsed 0.003950 s

$./sumMatrix 256 8
sumMatrixOnHost elapsed 0.318167 s
sumMatrixOnGPU2D <<<(64,2048), (256,8)>>> elapsed 0.000056 s
Error: sumMatrix.cu:200, code: 9, reason: invalid configuration argument

规律：
1. 最后一次的执行配置块的大小为（256,8），这是无效的，一个块中的线程总数超过了 1024 个 （GPU的硬件限制）

2. 最优：（128,4）

3.第一种情况，block 大小为（64,2），尽管在这种情况下启动的线程块最多，但不是最快的配置

4. 因为第三种情况中 block 的配置为（64,8），与最好的情况相比有相同数量的block
这两种情况应该在device上显示出相同的并行性。因为这种情况相比（128,4）仍然表现较差
结论:block最内层维度的大小对性能起到关键的作用

5.其他情况下，block的数量都比最小的情况少。增大并行性仍然是性能优化的一个重要因素

block 最少的那些示例应该显式出较低的可实现占用率，线程块最多的那些例子应该显示出较高的可实现占用率
可以使用achieved_occupancy指标来进行验证：

1.
==38338== NVPROF is profiling process 38338, command: ./sumMatrix 64 2
sumMatrixOnHost elapsed 0.309914 s
sumMatrixOnGPU2D <<<(256,8192), (64,2)>>> elapsed 0.020381 s
==38338== Profiling application: ./sumMatrix 64 2
==38338== Profiling result:
==38338== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.908657    0.908657    0.908657

2.
==41431== NVPROF is profiling process 41431, command: ./sumMatrix 64 4
sumMatrixOnHost elapsed 0.309634 s
sumMatrixOnGPU2D <<<(256,4096), (64,4)>>> elapsed 0.020718 s
==41431== Profiling application: ./sumMatrix 64 4
==41431== Profiling result:
==41431== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.916673    0.916673    0.916673

3.
==42884== NVPROF is profiling process 42884, command: ./sumMatrix 64 8
sumMatrixOnHost elapsed 0.308830 s
sumMatrixOnGPU2D <<<(256,2048), (64,8)>>> elapsed 0.024033 s
==42884== Profiling application: ./sumMatrix 64 8
==42884== Profiling result:
==42884== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.886029    0.886029    0.886029

4.
==48210== NVPROF is profiling process 48210, command: ./sumMatrix 128 2
sumMatrixOnHost elapsed 0.310703 s
sumMatrixOnGPU2D <<<(128,8192), (128,2)>>> elapsed 0.020440 s
==48210== Profiling application: ./sumMatrix 128 2
==48210== Profiling result:
==48210== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.917837    0.917837    0.917837

5.
==53313== NVPROF is profiling process 53313, command: ./sumMatrix 128 4
sumMatrixOnHost elapsed 0.310696 s
sumMatrixOnGPU2D <<<(128,4096), (128,4)>>> elapsed 0.020161 s
==53313== Profiling application: ./sumMatrix 128 4
==53313== Profiling result:
==53313== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.890197    0.890197    0.890197

6.
sudo nvprof --metrics achieved_occupancy ./sumMatrix 128 8
==57348== NVPROF is profiling process 57348, command: ./sumMatrix 128 8
sumMatrixOnHost elapsed 0.311311 s
sumMatrixOnGPU2D <<<(128,2048), (128,8)>>> elapsed 0.020588 s
==57348== Profiling application: ./sumMatrix 128 8
==57348== Profiling result:
==57348== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.847290    0.847290    0.847290

7.
sudo nvprof --metrics achieved_occupancy ./sumMatrix 256 2
==59312== NVPROF is profiling process 59312, command: ./sumMatrix 256 2
sumMatrixOnHost elapsed 0.311425 s
sumMatrixOnGPU2D <<<(64,8192), (256,2)>>> elapsed 0.020775 s
==59312== Profiling application: ./sumMatrix 256 2
==59312== Profiling result:
==59312== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.890508    0.890508    0.890508


8.
==65537== NVPROF is profiling process 65537, command: ./sumMatrix 256 4
sumMatrixOnHost elapsed 0.310690 s
sumMatrixOnGPU2D <<<(64,4096), (256,4)>>> elapsed 0.021396 s
==65537== Profiling application: ./sumMatrix 256 4
==65537== Profiling result:
==65537== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.847069    0.847069    0.847069


(128,4),(256,4):拥有最高的性能，在这种情况下，将block.y设置为1来增大inter-block的并行性
观察性能发生了什么变化，这使得每个block大小减少了，引起了更多的block被启动来处理相同数量的数据，这样做会产生以下结果：

./sumMatrix 128 1
sumMatrixOnHost elapsed 0.315013 s
sumMatrixOnGPU2D <<<(128,16384), (128,1)>>> elapsed 0.003948 s

./sumMatrix 128 4
sumMatrixOnHost elapsed 0.315690 s
sumMatrixOnGPU2D <<<(128,4096), (128,4)>>> elapsed 0.003930 s

./sumMatrix 256 4
sumMatrixOnHost elapsed 0.316812 s
sumMatrixOnGPU2D <<<(64,4096), (256,4)>>> elapsed 0.003950 s

./sumMatrix 256 1
sumMatrixOnHost elapsed 0.316765 s
sumMatrixOnGPU2D <<<(64,16384), (256,1)>>> elapsed 0.003973 s

(128,4): 仍然为最优配置
查看指标：

==53313== NVPROF is profiling process 53313, command: ./sumMatrix 128 4
sumMatrixOnHost elapsed 0.310696 s
sumMatrixOnGPU2D <<<(128,4096), (128,4)>>> elapsed 0.020161 s
==53313== Profiling application: ./sumMatrix 128 4
==53313== Profiling result:
==53313== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.890197    0.890197    0.890197

$sudo nvprof --metrics achieved_occupancy ./sumMatrix 128 1
==35397== NVPROF is profiling process 35397, command: ./sumMatrix 128 1
sumMatrixOnHost elapsed 0.310862 s
sumMatrixOnGPU2D <<<(128,16384), (128,1)>>> elapsed 0.020409 s
==35397== Profiling application: ./sumMatrix 128 1
==35397== Profiling result:
==35397== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.912415    0.912415    0.912415

$sudo nvprof --metrics achieved_occupancy ./sumMatrix 256 1
==126980== NVPROF is profiling process 126980, command: ./sumMatrix 256 1
sumMatrixOnHost elapsed 0.310752 s
sumMatrixOnGPU2D <<<(64,16384), (256,1)>>> elapsed 0.020370 s
==126980== Profiling application: ./sumMatrix 256 1
==126980== Profiling result:
==126980== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.923797    0.923797    0.923797


$sudo nvprof --metrics gld_throughput ./sumMatrix 128 4
==19936== NVPROF is profiling process 19936, command: ./sumMatrix 128 4
sumMatrixOnHost elapsed 0.310935 s
sumMatrixOnGPU2D <<<(128,4096), (128,4)>>> elapsed 0.015944 s
==19936== Profiling application: ./sumMatrix 128 4
==19936== Profiling result:
==19936== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  515.86GB/s  515.86GB/s  515.86GB/s

$sudo nvprof --metrics gld_throughput ./sumMatrix 128 1
==29198== NVPROF is profiling process 29198, command: ./sumMatrix 128 1
sumMatrixOnHost elapsed 0.310947 s
sumMatrixOnGPU2D <<<(128,16384), (128,1)>>> elapsed 0.015985 s
==29198== Profiling application: ./sumMatrix 128 1
==29198== Profiling result:
==29198== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  514.70GB/s  514.70GB/s  514.70GB/s

sudo nvprof --metrics gld_throughput ./sumMatrix 256 1
==128352== NVPROF is profiling process 128352, command: ./sumMatrix 256 1
sumMatrixOnHost elapsed 0.310773 s
sumMatrixOnGPU2D <<<(64,16384), (256,1)>>> elapsed 0.017528 s
==128352== Profiling application: ./sumMatrix 256 1
==128352== Profiling result:
==128352== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  513.64GB/s  513.64GB/s  513.64GB/s


sudo nvprof --metrics gld_efficiency ./sumMatrix 128 4
==5690== NVPROF is profiling process 5690, command: ./sumMatrix 128 4
sumMatrixOnHost elapsed 0.311382 s
==5690== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(128,4096), (128,4)>>> elapsed 0.139763 s
==5690== Profiling application: ./sumMatrix 128 4
==5690== Profiling result:
==5690== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%


$sudo nvprof --metrics gld_efficiency ./sumMatrix 128 1
==45717== NVPROF is profiling process 45717, command: ./sumMatrix 128 1
sumMatrixOnHost elapsed 0.310566 s
==45717== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(128,16384), (128,1)>>> elapsed 0.142677 s
==45717== Profiling application: ./sumMatrix 128 1
==45717== Profiling result:
==45717== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%


sudo nvprof --metrics gld_efficiency ./sumMatrix 256 1
==129339== NVPROF is profiling process 129339, command: ./sumMatrix 256 1
sumMatrixOnHost elapsed 0.311524 s
==129339== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrixOnGPU2D(float*, float*, float*, int, int)" (done)
sumMatrixOnGPU2D <<<(64,16384), (256,1)>>> elapsed 0.139841 s
==129339== Profiling application: ./sumMatrix 256 1
==129339== Profiling result:
==129339== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: sumMatrixOnGPU2D(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%


值得注意的是，最好的配置不具有最高的可实现占用率，相比于（32,16）也不具有最高的load throughput
从这些实验中可以推断出，没有一个单独的指标可以直接优化性能，
需要在几个相关的指标间寻找一个恰当的平衡来达到最佳的总体性能

指标与性能
1. 在大部分情况下，一个单独的指标不能产生最佳的性能
2. 与总体性能最直接相关的指标或事件取决于 kernel 代码的本质
3. 在相关的指标与事件之间寻求一个好的平衡，已实现最优性能
4. 从不同的角度查看 kernel 以寻求 相关指标间的平衡
5. grid/block启发式算法为性能调节提供了一个很好的起点。
 */