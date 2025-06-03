/**
4.3.5 性能调整
优化设备内存带宽利用率有两个指标：
1. 对齐以及合并内存访问，以减少带宽的浪费
2. 足够的并发内存操作，以隐藏内存延迟

我们已经学习了如何组织内存访问模式以实现对齐合并的内存访问。
这样做在设备 DRAM 和 SM 片上内存或寄存器之间能确保有效利用字节移动。

实现并发内存访问最大化是通过以下方式获得的：
1.增加每个线程中执行独立内存操作的数量
2. 对kernel 函数启动的执行配置进行实验，以充分体现每个 SM 的并行性。
*/

/**
4.3.5.1 展开技术
包含内存操作的展开循环增加了更独立的内存操作。

考虑之前的readSegment 示例。按如下的方式修改readOffset kernel 函数，
使得每个线程都执行 4 个独立的内存操作。
因为每个加载过程都是独立的，所以你可以调用更多的并发内存访问：
*/

/*
4.3.5.2 增大并行性
为了充分体现并行性，应该对一个kernel 函数启动的网格和block大小进行实验，以找到kernel 函数最佳的执行配置
运行测试代码，在offset=0（对齐访问）的情况下，搜索最优的block 大小
*/

#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void readOffsetUnroll4(float *A,float *B,float *C,const int n,int offset){
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;
    if (k < n) {
        C[i] = A[k] + B[k];
    }

    if (k + blockDim.x < n) {
        C[i + blockDim.x]     = A[k + blockDim.x]     + B[k + blockDim.x];
    }

    if (k + 2 * blockDim.x < n) {
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
    }

    if (k + 3 * blockDim.x < n) {
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }

}

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

__global__ void readOffset(float *A, float *B, float *C, const int n,
                           int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffsetUnroll2(float *A, float *B, float *C, const int n,
                                  int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
    if (k + blockDim.x < n) {
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size
    int power = 20;
    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset       = atoi(argv[1]);
    if (argc > 2) blocksize    = atoi(argv[2]);
    if (argc > 3) power        = atoi(argv[3]);

    int nElem = 1 << power; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    //  summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    //  kernel 1:
    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup     <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 1
    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
            block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem-offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 2
    iStart = seconds();
    readOffsetUnroll2<<<grid.x/2, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("unroll2    <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
            grid.x / 2, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

    // kernel 3
    iStart = seconds();
    readOffsetUnroll4<<<grid.x / 4, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("unroll4    <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
            grid.x / 4, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);
    CHECK(cudaMemset(d_C, 0x00, nBytes));

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

/**
展开技术对性能有非常好的影响
在一些条件下，展开可以获得比地址对齐还要好的优化。
相比于无循环展开的readSegment，循环展开技术有可能获得加速（但在高算力的GPU 上有限，如下）：
$./readSegmentUnroll 0
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
warmup     <<< 2048,  512 >>> offset    0 elapsed 0.000129 sec
readOffset <<< 2048,  512 >>> offset    0 elapsed 0.000040 sec
unroll2    <<< 1024,  512 >>> offset    0 elapsed 0.000052 sec
unroll4    <<<  512,  512 >>> offset    0 elapsed 0.000051 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 11
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
warmup     <<< 2048,  512 >>> offset   11 elapsed 0.000129 sec
readOffset <<< 2048,  512 >>> offset   11 elapsed 0.000037 sec
unroll2    <<< 1024,  512 >>> offset   11 elapsed 0.000048 sec
unroll4    <<<  512,  512 >>> offset   11 elapsed 0.000047 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 128
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
warmup     <<< 2048,  512 >>> offset  128 elapsed 0.000122 sec
readOffset <<< 2048,  512 >>> offset  128 elapsed 0.000037 sec
unroll2    <<< 1024,  512 >>> offset  128 elapsed 0.000048 sec
unroll4    <<<  512,  512 >>> offset  128 elapsed 0.000047 sec

对于 I/O 密集型的kernel，内存访问并行有很高的优先级

注意，展开并不影响执行内存操作的数量（只影响并发执行的数量）
可以通过使用以下指令来测试非对齐情况下（offset=11),测量原始kernel函数和展开kernel 函数的负载和存储效率指标：
==53973== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: readOffsetUnroll2(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%
          1                          gld_transactions                  Global Load Transactions      327676      327676      327676
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131071      131071      131071
    Kernel: readOffsetUnroll4(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%
          1                          gld_transactions                  Global Load Transactions      327676      327676      327676
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131071      131071      131071
    Kernel: readOffset(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%
          1                          gld_transactions                  Global Load Transactions      327676      327676      327676
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131071      131071      131071
    Kernel: warmup(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%
          1                          gld_transactions                  Global Load Transactions      327676      327676      327676
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131071      131071      131071

load/store 的效率以及transaction 的数量均相同
*/

/**
并行性实验（块大小实验）
[./readSegmentUnroll 0 1024 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 4096, 1024 >>> offset    0 elapsed 0.000171 sec
readOffset <<< 4096, 1024 >>> offset    0 elapsed 0.000094 sec
unroll2    <<< 2048, 1024 >>> offset    0 elapsed 0.000106 sec
unroll4    <<< 1024, 1024 >>> offset    0 elapsed 0.000104 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 0 512 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 8192,  512 >>> offset    0 elapsed 0.000173 sec
readOffset <<< 8192,  512 >>> offset    0 elapsed 0.000093 sec
unroll2    <<< 4096,  512 >>> offset    0 elapsed 0.000098 sec
unroll4    <<< 2048,  512 >>> offset    0 elapsed 0.000099 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 0 256 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 16384,  256 >>> offset    0 elapsed 0.000168 sec
readOffset <<< 16384,  256 >>> offset    0 elapsed 0.000093 sec
unroll2    <<< 8192,  256 >>> offset    0 elapsed 0.000105 sec
unroll4    <<< 4096,  256 >>> offset    0 elapsed 0.000112 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 0 128 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 32768,  128 >>> offset    0 elapsed 0.000174 sec
readOffset <<< 32768,  128 >>> offset    0 elapsed 0.000092 sec
unroll2    <<< 16384,  128 >>> offset    0 elapsed 0.000099 sec
unroll4    <<< 8192,  128 >>> offset    0 elapsed 0.000102 sec

512 时性能最佳
相比于1024时，线程块数量增加了。
尽管 每个block 128 线程可以增加并行性，但是因为并发性硬性限制：
如 Fermi GPU 中，每个 SM 最多有8个并发block
每个 SM 最多 48个 并发wrap
同时过低的thread 数量可能导致无法充分利用资源：例如在上述配置中，若128个线程，则只有4个块，因此一个 SM最多运行32个wrap，无法跑满 48个并发wrap，算力发挥不足（可以使用CUDA 占用率验证）
非对齐情况：
$./readSegmentUnroll 11 1024 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 4096, 1024 >>> offset   11 elapsed 0.000168 sec
readOffset <<< 4096, 1024 >>> offset   11 elapsed 0.000093 sec
unroll2    <<< 2048, 1024 >>> offset   11 elapsed 0.000102 sec
unroll4    <<< 1024, 1024 >>> offset   11 elapsed 0.000111 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 11 512 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 8192,  512 >>> offset   11 elapsed 0.000165 sec
readOffset <<< 8192,  512 >>> offset   11 elapsed 0.000093 sec
unroll2    <<< 4096,  512 >>> offset   11 elapsed 0.000099 sec
unroll4    <<< 2048,  512 >>> offset   11 elapsed 0.000104 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 11 256 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 16384,  256 >>> offset   11 elapsed 0.000176 sec
readOffset <<< 16384,  256 >>> offset   11 elapsed 0.000093 sec
unroll2    <<< 8192,  256 >>> offset   11 elapsed 0.000101 sec
unroll4    <<< 4096,  256 >>> offset   11 elapsed 0.000105 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./readSegmentUnroll 11 128 22
./readSegmentUnroll starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 4194304
warmup     <<< 32768,  128 >>> offset   11 elapsed 0.000173 sec
readOffset <<< 32768,  128 >>> offset   11 elapsed 0.000093 sec
unroll2    <<< 16384,  128 >>> offset   11 elapsed 0.000095 sec
unroll4    <<< 8192,  128 >>> offset   11 elapsed 0.000101 sec

128 时性能最佳，这代表着此时增加并发性对性能的增益。

最大化带宽利用率：
影响设备内存操作性能的因素主要有两个：
1. 有效利用 设备 DRAM 和 SM 片上内存之间的字节移动：为了避免设备内存带宽的浪费，内存访问模式应该是对齐和合并的。
2. 当前的并发内存操作数：可以通过两种方式最大化当前存储器的操作数：1）展开，每个线程产生更多的独立内存访问；
2）修改kernel 函数启动的执行配置来使每个 SM 有更多的并行性。
*/

