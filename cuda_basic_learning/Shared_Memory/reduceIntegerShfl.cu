#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define DIM     128
#define SMEMDIM 8     // 128/32 = 8 

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

__global__ void reduceSmem (int *g_idata, int *g_odata, unsigned int n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}


/*
5.6.3 使用线程束洗牌指令的并行归约
在前面的 5.3.1 节中，已经介绍了如何使用共享内存来优化并行归约算法。
在本节中，将介绍如何使用线程束洗牌指令来解决相同的问题。
基本思路非常简单，它包括 3 个层面的归纳：
线程束级归约
线程块级归约
网格级归约

一个线程块中可能有几个线程束。
对于线程束级归约来说，每个线程束执行自己的归约。
每个线程不使用共享内存，而是使用寄存器存储从一个全局内存中读取的数据元素。
*/

// __inline__ __device__ int warpReduce(int localSum){
//     localSum += __shfl_xor(localSum,16);
//     localSum += __shfl_xor(localSum,8);
//     localSum += __shfl_xor(localSum,4);
//     localSum += __shfl_xor(localSum,2);
//     localSum += __shfl_xor(localSum,1);
//     return localSum;
// }
__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);

    return localSum;
}

/*
在上述的函数返回之后，每个线程束的总和保存到基于线程索引和线程束大小的共享内存中：
int laneIdx = threadIdx.x % warpSize;
int warpIdx = threadIdx.x / warpSize;
mySum = warpReduce(mySum);
if (laneIdx == 0){
    smem[warpIdx] = mySum;
}

对于线程块级归约，先同步块，然后使用相同的线程束归约函数将每个线程束的总和进行相加。
之后，由线程块产生的最终输出由块中的第一个线程保存到全局内存中，如下所示：
__syncthreads();
mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx]:0;
if(warpIdx == 0) mySum = warpReduce(mySum);
if(threadIdx.x == 0) g_odata[blockIdx.x] = mySum;

对于网格级归约，g_odata 被复制回到执行最终归约的主机中。下面是完整的reduceShfl的kernel函数：
*/
__global__ void reduceShfl (int *g_idata, int *g_odata, unsigned int n,int warpSize)
{
    // shared memory for each warp sum
    extern __shared__ int smem[];

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // blcok-wide warp reduce
    int localSum = warpReduce(g_idata[idx]);

    // save warp sum to shared memory
    if (laneIdx == 0) smem[warpIdx] = localSum;

    // block synchronization
    __syncthreads();

    // last warp reduce
    if (threadIdx.x < warpSize) localSum = (threadIdx.x < SMEMDIM) ?
        smem[laneIdx] : 0;

    if (warpIdx == 0) localSum = warpReduce(localSum);

    // write result for this block to global mem
    if (threadIdx.x == 0) g_odata[blockIdx.x] = localSum;
}

// __global__ void reduceShfl(int *g_idata,int *g_odata,unsigned int n,int warpSize){
//     //shared memory for each warp sum
//     // blockDim.x 
//     int tmpSize = (blockDim.x/warpSize);
//     extern __shared__ int smem[];

//     //boundary check
//     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if(idx>=n){
//         return;
//     }

//     //calculate lane index and warp index
//     int laneIdx = threadIdx.x % warpSize;
//     int warpIdx = threadIdx.x / warpSize;

//     //block-wide warp reduce
//     int localSum = warpReduce(g_idata[idx]);

//     //save warp sum to shared memory
//     if(laneIdx == 0){
//         smem[warpIdx] = localSum;
//     }

//     //block synchronization
//     __syncthreads();
    
//     // last warp reduce
//     if (threadIdx.x < warpSize) localSum = (threadIdx.x < SMEMDIM) ?
//         smem[laneIdx] : 0;

//     if (warpIdx == 0) localSum = warpReduce(localSum);

//     // write result for this block to global mem
//     if (threadIdx.x == 0) g_odata[blockIdx.x] = localSum;
// }

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int ishift = 10;

    if(argc > 1) ishift = atoi(argv[1]);

    int size = 1 << ishift;
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = DIM;   // initial block size

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
        h_idata[i] = (int)( rand() & 0xFF );

    memcpy (tmp, h_idata, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    int cpu_sum = recursiveReduce (tmp, size);
    printf("cpu reduce          : %d\n", cpu_sum);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceSmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceShfl<<<grid.x, block,block.x/32*sizeof(int)>>>(d_idata, d_odata, size,32);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceShfl          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}

/*
性能如下：
==116981== Profiling application: ./reduceIntegerShfl
==116981== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   25.93%  3.7760us         1  3.7760us  3.7760us  3.7760us  reduceSmem(int*, int*, unsigned int)
                   25.49%  3.7120us         2  1.8560us  1.7920us  1.9200us  [CUDA memcpy HtoD]
                   24.84%  3.6160us         1  3.6160us  3.6160us  3.6160us  reduceShfl(int*, int*, unsigned int, int)
                   23.74%  3.4560us         2  1.7280us  1.4720us  1.9840us  [CUDA memcpy DtoH]
*/

