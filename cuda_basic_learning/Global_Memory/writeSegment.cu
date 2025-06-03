/*
4.3.3 global memory store/write 全局内存写入

内存的store 操作相对简单。L1 Cache 不能用在 Fermi 或 Kepler GPU 上进行存储操作，
在发送到 deivce memory 之前存储操作只通过 L2 Cache。store 操作在32个字节段的粒度上被执行。
内存 transaction 可以同时被分为一段、两端或四段。
例如，如果两个地址同属于一个128字节区域，但不属于一个对齐的64字节区域，则会执行一个四段事务
（也就是说，执行一个四段事务比执行两个一？两？段事务效果更好）。

1. 理想情况：内存访问是对齐的，并且wrap里所有的thread访问一个连续的128字节范围。存储请求由一个四段事务实现：
                                                     '          '          '
wrap 中的地址：                           | | | | | | '| | | | | '| | | | | '| | | | | |
                                         v v v v v v 'v v v v v 'v v v v v 'v v v v v v
0--------32--------64--------96--------128--------160'-------192'-------224'--------256--------288--------320
（一个四段传输）

2. 内存访问是对齐的，但地址分散在一个192字节范围内的情况。存储可以通过3个一段事务完成：

                                                                         
wrap 中的地址：              | | | | | |              | | | | | |                     | | | | | |
                            v v v v v v              v v v v v v                     v v v v v v
0--------32--------64--------96--------128--------160---------192-------224--------256--------288--------320
（三个一段传输）

3. 内存访问是对齐的，并且地址访问在一个连续的64个字节范围内的情况：存储请求由一个两段事务来完成：
                                                     '          '          
wrap 中的地址：                           | | | | | | '| | | | | '
                                         v v v v v v 'v v v v v '
0--------32--------64--------96--------128--------160'-------192'-------224'--------256--------288--------320
（一个两段传输）
*/

/*
非对齐写入的示例
为了验证非对齐对内存存储效率的影响，按照下面的方式修改向量加法kernel 函数。仍然使用两个不同的索引：
索引 k 根据给定的 offset 进行变化，而索引 i 不变（并因此产生对齐访问）
使用对齐索引 i 从数组 A 和数组 B进行加载，以产生良好的内存加载效率。使用偏移量索引 x 写入输入 C，
可能会造成非对齐写入，这取决于偏移量的值：
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void writeOffset(float *A,float *B,float *C,const int n, int offset){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if(k < n){
        C[k] = A[i] + B[i];
    }
}

//按以上要求修改主机端向量加法代码：
void checkResult(float *hostRef, float *gpuRef, const int N, const int offset)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = offset; i < N; i++)
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
        C[idx] = A[k] + B[k];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[k] = A[i] + B[i];
}

__global__ void writeOffsetUnroll2(float *A, float *B, float *C, const int n,
                                   int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k + blockDim.x < n)
    {
        C[k]            = A[i]            + B[i];
        C[k + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
    }
}

__global__ void writeOffsetUnroll4(float *A, float *B, float *C, const int n,
                                   int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k + 3 * blockDim.x < n)
    {
        C[k]              = A[i]              + B[i];
        C[k + blockDim.x]   = A[i +  blockDim.x] + B[i +  blockDim.x];
        C[k + 2 * blockDim.x] = A[i + 2 * blockDim.x] + B[i + 2 * blockDim.x];
        C[k + 3 * blockDim.x] = A[i + 3 * blockDim.x] + B[i + 3 * blockDim.x];
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
    int nElem = 1 << 20; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset    = atoi(argv[1]);

    if (argc > 2) blocksize = atoi(argv[2]);

    // execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup
    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup      <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // kernel 1:
    iStart = seconds();
    writeOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("writeOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem, offset);

    // // kernel 2
    // iStart = seconds();
    // writeOffsetUnroll2<<<grid.x / 2, block>>>(d_A, d_B, d_C, nElem / 2, offset);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;
    // printf("unroll2     <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
    //         grid.x / 2, block.x, offset, iElaps);
    // CHECK(cudaGetLastError());

    // // copy kernel result back to host side and check device results
    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    // checkResult(hostRef, gpuRef, nElem, offset);

    // // kernel 2
    // iStart = seconds();
    // writeOffsetUnroll4<<<grid.x / 4, block>>>(d_A, d_B, d_C, nElem / 2, offset);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;
    // printf("unroll4     <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
    //         grid.x / 4, block.x, offset, iElaps);
    // CHECK(cudaGetLastError());

    // // copy kernel result back to host side and check device results
    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    // checkResult(hostRef, gpuRef, nElem, offset);

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
$./writeSegment 0
./writeSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
warmup      <<< 2048,  512 >>> offset    0 elapsed 0.000126 sec
writeOffset <<< 2048,  512 >>> offset    0 elapsed 0.000041 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./writeSegment 11
./writeSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
warmup      <<< 2048,  512 >>> offset   11 elapsed 0.000127 sec
writeOffset <<< 2048,  512 >>> offset   11 elapsed 0.000047 sec
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./writeSegment 128
./writeSegment starting reduction at device 0: Tesla V100-SXM2-16GB  with array size 1048576
warmup      <<< 2048,  512 >>> offset  128 elapsed 0.000124 sec
writeOffset <<< 2048,  512 >>> offset  128 elapsed 0.000044 sec

offset = 11 性能最差，非对齐，使用nvprof 获取全局加载和存储效率指标，可以发现这一非对齐情况的原因：

offset = 0 Global Memory Store Efficiency      100.00%
sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency,gst_transactions ./writeSegment 0

warmup      <<< 2048,  512 >>> offset    0 elapsed 0.032930 sec
writeOffset <<< 2048,  512 >>> offset    0 elapsed 0.017461 sec

==1860== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: writeOffset(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131072      131072      131072
    Kernel: warmup(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131072      131072      131072

offset = 128 Global Memory Store Efficiency      100.00%
$sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency,gst_transactions ./writeSegment 128
warmup      <<< 2048,  512 >>> offset  128 elapsed 0.037282 sec
writeOffset <<< 2048,  512 >>> offset  128 elapsed 0.017522 sec

==4454== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: writeOffset(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131056      131056      131056
    Kernel: warmup(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
          1                          gst_transactions                 Global Store Transactions      131056      131056      131056

offset = 11 (Global Memory Store Efficiency      80.00%)
$sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency,gst_transactions ./writeSegment 11

warmup      <<< 2048,  512 >>> offset   11 elapsed 0.037304 sec
writeOffset <<< 2048,  512 >>> offset   11 elapsed 0.016138 sec

==9996== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: writeOffset(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency      80.00%      80.00%      80.00%
          1                          gst_transactions                 Global Store Transactions      163838      163838      163838
    Kernel: warmup(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency      80.00%      80.00%      80.00%
          1                          gst_transactions                 Global Store Transactions      163838      163838      163838

除了非对齐的情况（offset=11）的store，所有load 和store的效率均为100%。非对齐写入的存储效率为80%
当偏移量为11 且从一个wrap 产生一个128个字节的写入请求时，该请求由一个四段事务和一个一个一段事务来实现。
因此，128个字节用来请求，160个字节用来写入，存储效率为80%
*/

