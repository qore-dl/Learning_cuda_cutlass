#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*
使用宏定义，将块大小设置为恒定的128个线程：
*/
#define DIM 128

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}


/*
5.3 减少全局内存访问
使用共享内存的主要原因之一是要缓存片上的数据，从而减少kernel函数中全局内存访问的次数。
第3章介绍了使用全局内存的并行归约kernel函数，并集中解释了以下几个问题：

1.如何重新安排数据访问模式以避免线程束分化。
2.如何展开循环以保证有足够的操作使得指令和内存带宽饱和。

在本节中，将重新使用这些并行归约kernel函数，但是这里使用共享内存作为可编程管理缓存以减少全局内存的访问。

5.3.1 使用共享内存的并行归约
下面的 reduceGmem kernel 函数将被用作基准性能的起点，在第3章中介绍过该函数。
实现并行归约只使用全局内存，输入元素的内循环是完全展开的。 
*/

__global__ void reduceGmem(int *g_idata,int *g_odata,unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    /*
    这个kernel 函数有4个主要部分。首先，计算数据块的偏移量，该数据块的属于线程块，与全局输入有关。
    */
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n){
        return;
    }
    /*
    接下来，kernel 函数执行一个使用全局内存的原地归约，将其归约到32个元素
    */
    //in-place reduction in global memory
    if(blockDim.x >= 1024 && tid < 512){
        idata[tid] += idata[tid + 512];
    }

    __syncthreads();

    if(blockDim.x >= 512 && tid < 256){
        idata[tid] += idata[tid+256];
    }

    __syncthreads();

    if(blockDim.x >= 256 && tid < 128){
        idata[tid] += idata[tid+128];
    }

    __syncthreads();

    if(blockDim.x >= 128 && tid < 64){
        idata[tid] += idata[tid+64];
    }

    __syncthreads();
    /*
    然后，kernel 函数执行原地归约，这个过程仅使用每个线程块的第一个wrap。
    注意，在循环展开的部分，volatile 修饰符用来确保当线程束在锁步中执行时，只有最新数值能被读取
    因为这是直接从全局内存中读取，不会经过寄存器、缓存等。
    */
    //unrolling warp
    if (tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    //write result for this block to global mem
    /*
    最后，分配给该线程块的输入数据块总数被写回到全局内存中
    */
    if(tid==0){
        g_odata[blockIdx.x] = idata[0];
    }

}

/*
接下来，考虑下面的原地归约kernel函数 reduceSmem，它增加了带有共享内存的全局内存操作。
这个kernel函数和原来的reduceGmem kernel 函数机会相同。
然而，reduceSmem 函数没有使用全局内存中的输入数组子集来执行原地归约，而是使用了共享内存数组
smem。smem 被声明为与每个线程块具有相同的维度：__shared__ int smem[DIM];
每个线程块都用它的全局输入数据块来初始化smem 数组：
smem[tid] = idata[tid];
__syncthreads();

然后原地归约是使用共享内存（smem）被执行的，而不是使用全局内存（idata）。reduceSmem kernel 函数的代码如下：
*/

__global__ void reduceSmem(int *g_idata,int *g_odata,unsigned int n){
    __shared__ int smem[DIM];
     
    //set shared ID
    unsigned int tid = threadIdx.x;

    //boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n){
        return;
    }

    //convert global data pointer to the local pointer of this block:
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //set to smem by each threads:
    smem[tid] = idata[tid];
    __syncthreads();

    //in-place reduction in shared memory
    if(blockDim.x>=1024 && tid<512){
        smem[tid] += smem[tid+512];
    }
    __syncthreads();

    if(blockDim.x>=512 && tid<256){
        smem[tid] += smem[tid+256];
    }
    __syncthreads();

    if(blockDim.x>=256 && tid<128){
        smem[tid] += smem[tid+128];
    }
    __syncthreads();

    if(blockDim.x>=128 && tid<64){
        smem[tid] += smem[tid+64];
    }
    __syncthreads();

    //unrolling warp
    if(tid<32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid+32];
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8];
        vsmem[tid] += vsmem[tid+4];
        vsmem[tid] += vsmem[tid+2];
        vsmem[tid] += vsmem[tid+1];
    }
    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = smem[0];
    }
}

// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmemUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx < n)
    {
        int a1, a2, a3, a4;
        a1 = a2 = a3 = a4 = 0;
        a1 = g_idata[idx];
        if (idx + blockDim.x < n) a2 = g_idata[idx + blockDim.x];
        if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
        if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/*
5.3.2 使用展开的并行归约
在前面的kernel函数中，每个线程块处理一个数据块。在第3章中，可以通过一次运行多个 I/O 操作，
展开线程块来提高内核性能。以下内核展开了4个线程块，即每个线程处理来自于 4 个数据块的数据元素。
通过展开，以下优势是可预期的：
1. 通过在每个线程中提供更多的并行 I/O，增加全局内存的吞吐量
2. 全局内存存储事务减少了1/4
3. 整体内核性能的提升。
kernel 函数代码如下：
*/
__global__ void reduceSmemUnroll(int *g_idata,int *g_odata,unsigned int n){
    //static shared memory
    __shared__ int smem[DIM];

    //set thread ID
    unsigned int tid = threadIdx.x;
    
    //global index, 4 blocks of input data processed at a time
    /*
    要使每个线程处理 4 个数据元素，第一步是基于每个线程的线程块和线程索引，重新计算全局输入数据的偏移：
    */
    unsigned int idx = blockIdx.x * blockDim.x *4 + threadIdx.x;

    //unrolling 4 blocks
    int tmpSum = 0;

    //boundary check
    /*
    因为每个线程读取 4 个数据元素，所以每个线程的处理起点现在被偏移为就好像是线程块的4倍
    利用这个新的偏移，每个线程读取 4 个数据元素，然后将其添加到局部变量tmpSum中。
    然后，tmpSum 用于初始化共享内存，而不是直接从全局内存进行初始化。
    */
    if (idx < n){
        int a1,a2,a3,a4;
        a1 = a2 = a3 = a4=0;
        a1 = g_idata[idx];
        if(idx+blockDim.x<n){
            a2 = g_idata[idx+blockDim.x];
        }
        if(idx+blockDim.x*2<n){
            a3 = g_idata[idx+blockDim.x*2];
        }
        if(idx+blockDim.x*3<n){
            a4 = g_idata[idx+blockDim.x*3];
        }
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in shared memory
    if(blockDim.x>=1024 && tid < 512){
        smem[tid] += smem[tid+512];
    }
    __syncthreads();

    if(blockDim.x>=512 && tid < 256){
        smem[tid] += smem[tid+256];
    }
    __syncthreads();
    
    if(blockDim.x>=256 && tid < 128){
        smem[tid] += smem[tid+128];
    }
    __syncthreads();

    if(blockDim.x>=128 && tid < 64){
        smem[tid] += smem[tid+64];
    }
    __syncthreads();

    //unrolling wrap
    if(tid < 32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if(tid==0){
        g_odata[blockIdx.x] = smem[0];
    }
}

/*
5.3.3 使用动态共享内存的并行归约
并行归约kernel函数还可以使用动态共享内存来执行，通过一下声明，在reduceSmemUnroll中使用动态共享内存取代静态共享内存：
extern __shared__ int smem[];
*/
__global__ void reduceSmemDyn(int *g_idata,int *g_odata,unsigned int n){
    //dynamic shared memory
    extern __shared__ int smem[];
     
    //set shared ID
    unsigned int tid = threadIdx.x;

    //boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n){
        return;
    }

    //convert global data pointer to the local pointer of this block:
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //set to smem by each threads:
    smem[tid] = idata[tid];
    __syncthreads();

    //in-place reduction in shared memory
    if(blockDim.x>=1024 && tid<512){
        smem[tid] += smem[tid+512];
    }
    __syncthreads();

    if(blockDim.x>=512 && tid<256){
        smem[tid] += smem[tid+256];
    }
    __syncthreads();

    if(blockDim.x>=256 && tid<128){
        smem[tid] += smem[tid+128];
    }
    __syncthreads();

    if(blockDim.x>=128 && tid<64){
        smem[tid] += smem[tid+64];
    }
    __syncthreads();

    //unrolling warp
    if(tid<32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid+32];
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8];
        vsmem[tid] += vsmem[tid+4];
        vsmem[tid] += vsmem[tid+2];
        vsmem[tid] += vsmem[tid+1];
    }
    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = smem[0];
    }
}

__global__ void reduceSmemUnrollDyn(int *g_idata,int *g_odata,unsigned int n){
    //dynamic shared memory
    extern __shared__ int smem[];

    //set thread ID
    unsigned int tid = threadIdx.x;
    
    //global index, 4 blocks of input data processed at a time
    /*
    要使每个线程处理 4 个数据元素，第一步是基于每个线程的线程块和线程索引，重新计算全局输入数据的偏移：
    */
    unsigned int idx = blockIdx.x * blockDim.x *4 + threadIdx.x;

    //unrolling 4 blocks
    int tmpSum = 0;

    //boundary check
    /*
    因为每个线程读取 4 个数据元素，所以每个线程的处理起点现在被偏移为就好像是线程块的4倍
    利用这个新的偏移，每个线程读取 4 个数据元素，然后将其添加到局部变量tmpSum中。
    然后，tmpSum 用于初始化共享内存，而不是直接从全局内存进行初始化。
    */
    if (idx < n){
        int a1,a2,a3,a4;
        a1 = a2 = a3 = a4=0;
        a1 = g_idata[idx];
        if(idx+blockDim.x<n){
            a2 = g_idata[idx+blockDim.x];
        }
        if(idx+blockDim.x*2<n){
            a3 = g_idata[idx+blockDim.x*2];
        }
        if(idx+blockDim.x*3<n){
            a4 = g_idata[idx+blockDim.x*3];
        }
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in shared memory
    if(blockDim.x>=1024 && tid < 512){
        smem[tid] += smem[tid+512];
    }
    __syncthreads();

    if(blockDim.x>=512 && tid < 256){
        smem[tid] += smem[tid+256];
    }
    __syncthreads();
    
    if(blockDim.x>=256 && tid < 128){
        smem[tid] += smem[tid+128];
    }
    __syncthreads();

    if(blockDim.x>=128 && tid < 64){
        smem[tid] += smem[tid+64];
    }
    __syncthreads();

    //unrolling wrap
    if(tid < 32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if(tid==0){
        g_odata[blockIdx.x] = smem[0];
    }
}

/*
5.3.4 有效带宽
由于归约 kernel 函数是受内存带宽约束的，所以评估它们时所使用的适当的性能指标是有效带宽。
有效带宽是在kernel函数的完整执行时间内 I/O 的数量（以字节为单位）。
对于内存约束的应用程序，有效带宽是一个估算实际带宽利用率的很好的指标。它可以表示为：
有效带宽 = （读字节+写字节）/ (运行时间x10^9) GB/s
*/

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
    /*
    在所有的测试中，使用以下的语句，将数组的长度设置为 16M，这相当于减少了整型的数目
    */
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    /*
    将块大小设置为恒定的128个线程：
    */
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
    {
        h_idata[i] = (int)( rand() & 0xFF );
    }

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

    // // reduce gmem
    // CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    // reduceNeighboredGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    // CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
    //                  cudaMemcpyDeviceToHost));
    // gpu_sum = 0;

    // for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    // printf("reduceNeighboredGmem: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
    //        block.x);

    // // reduce gmem
    // CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    // reduceNeighboredSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    // CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
    //                  cudaMemcpyDeviceToHost));
    // gpu_sum = 0;

    // for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    // printf("reduceNeighboredSmem: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
    //        block.x);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceGmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

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
    reduceSmemDyn<<<grid.x, block, blocksize*sizeof(int)>>>(d_idata, d_odata,
            size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceSmemDyn       : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

    

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceGmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("reduceGmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum,
            grid.x / 4, block.x);

    // reduce smem

    /*
    在这个展开下，全局内存加载事务的数量在kernel 函数中没有变化，但是全局内存存储事务的数量减少了1/4
    此外，一次运行4个全局加载运算， GPU 在并发调度时有了更大的灵活性，因此可能会产生更好的全局内存利用率。

    */
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("reduceSmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum,
            grid.x / 4, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    // reduceSmemUnrollDyn<<<grid.x / 4, block, DIM*sizeof(int)>>>(d_idata,
            // d_odata, size);
    // 启动kernel 函数时，必须指定待动态分配的共享内存数量：
    reduceSmemUnrollDyn<<<grid.x/4,block,DIM*sizeof(int)>>>(d_idata,d_odata,size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("reduceSmemDynUnroll4: %d <<<grid %d block %d>>>\n", gpu_sum,
            grid.x / 4, block.x);

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
使用全局内存进行归约的kernel 函数的性能如下：
$nvprof ./reduceInteger
==89679== NVPROF is profiling process 89679, command: ./reduceInteger
./reduceInteger starting reduction at device 0: Tesla V100-SXM2-16GB     with array size 16777216  grid 131072 block 128
cpu reduce          : 2139353471
reduceGmem          : 2139353471 <<<grid 131072 block 128>>>
==89679== Profiling application: ./reduceInteger
==89679== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.53%  12.125ms         1  12.125ms  12.125ms  12.125ms  [CUDA memcpy HtoD]
                    1.96%  243.07us         1  243.07us  243.07us  243.07us  reduceGmem(int*, int*, unsigned int)

使用全局内存和使用共享内存进行归约的kernel 函数的性能如下：
==34149== Profiling application: ./reduceInteger
==34149== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.57%  24.034ms         2  12.017ms  11.879ms  12.155ms  [CUDA memcpy HtoD]
                    0.98%  242.27us         1  242.27us  242.27us  242.27us  reduceGmem(int*, int*, unsigned int)
                    0.90%  220.67us         1  220.67us  220.67us  220.67us  reduceSmem(int*, int*, unsigned int)
                    0.55%  136.42us         2  68.208us  67.456us  68.960us  [CUDA memcpy DtoH]
性能有了一定的提升，使用下列指标测试全局内存加载和存储事务，看一下共享内存是如何很好地减少全局内存访问的：

gld_transactions: 全局内存load事务的数量
gst_transactions: 全局内存store事务的数量

结果总结如下：
==40031== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceSmem(int*, int*, unsigned int)
          1                          gld_transactions                  Global Load Transactions     2097152     2097152     2097152
          1                          gst_transactions                 Global Store Transactions      131072      131072      131072
    Kernel: reduceGmem(int*, int*, unsigned int)
          1                          gld_transactions                  Global Load Transactions     8912896     8912896     8912896
          1                          gst_transactions                 Global Store Transactions     4325376     4325376     4325376
可以看出， 使用共享内存后，load 操作减少 4.25x 而store 操作则减少了33x
因此，使用共享内存明显减少了全局内存访问。

比较在全局内存和共享内存条件下使用展开技术的情况：
==11223== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.29%  47.893ms         4  11.973ms  11.850ms  12.194ms  [CUDA memcpy HtoD]
                    0.50%  242.62us         1  242.62us  242.62us  242.62us  reduceGmem(int*, int*, unsigned int)
                    0.45%  220.61us         1  220.61us  220.61us  220.61us  reduceSmem(int*, int*, unsigned int)
                    0.33%  162.69us         4  40.671us  14.080us  68.448us  [CUDA memcpy DtoH]
                    0.24%  116.38us         1  116.38us  116.38us  116.38us  reduceGmemUnroll(int*, int*, unsigned int)
                    0.18%  89.664us         1  89.664us  89.664us  89.664us  reduceSmemUnroll(int*, int*, unsigned int)
共享内存展开相比于全局内存展开而言，性能明显提升。
在这个展开下，全局内存加载事务的数量在kernel 函数中没有变化，但是全局内存存储事务的数量减少了1/4
此外，一次运行4个全局加载运算， GPU 在并发调度时有了更大的灵活性，因此可能会产生更好的全局内存利用率。

有了这些变化，被 4 展开的共享内存kernel 函数（reduceSmemUnroll）比最原始的共享内存函数（reduceSmem）快了 2.5 倍
使用nvprof 检查全局内存事务后，同reduceSmem 相比，reduceSmemUnroll 函数中存储事务的数量减少了1/4，
然而加载事务的数量保持不变，结果如下所示：
==11689== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: reduceSmem(int*, int*, unsigned int)
gld_transactions                  Global Load Transactions     2097152     2097152     2097152
gst_transactions                 Global Store Transactions      131072      131072      131072
Kernel: reduceSmemUnroll(int*, int*, unsigned int)
gld_transactions                  Global Load Transactions     2097152     2097152     2097152
gst_transactions                 Global Store Transactions       32768       32768       32768
Kernel: reduceGmem(int*, int*, unsigned int)
gld_transactions                  Global Load Transactions     8912896     8912896     8912896
gst_transactions                 Global Store Transactions     4325376     4325376     4325376
Kernel: reduceGmemUnroll(int*, int*, unsigned int)
gld_transactions                  Global Load Transactions     4325376     4325376     4325376
gst_transactions                 Global Store Transactions     1605632     1605632     1605632
最后，检查全局内存吞吐量。加载吞吐量增加了2.68倍，而存储吞吐量下降了1.49倍。
加载吞吐量的增加归因于大量的同时加载请求。存储吞吐量的下降是因为较少的存储请求使总线没有达到饱和。结果总结如下：
==52782== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: reduceSmem(int*, int*, unsigned int)
gld_throughput                    Global Load Throughput  285.44GB/s  285.44GB/s  285.44GB/s
gst_throughput                   Global Store Throughput  17.840GB/s  17.840GB/s  17.840GB/s
Kernel: reduceSmemUnroll(int*, int*, unsigned int)
gld_throughput                    Global Load Throughput  767.02GB/s  767.02GB/s  767.02GB/s
gst_throughput                   Global Store Throughput  11.985GB/s  11.985GB/s  11.985GB/s
Kernel: reduceGmem(int*, int*, unsigned int)
gld_throughput                    Global Load Throughput  1127.7GB/s  1127.7GB/s  1127.7GB/s
gst_throughput                   Global Store Throughput  547.25GB/s  547.25GB/s  547.25GB/s
Kernel: reduceGmemUnroll(int*, int*, unsigned int)
gld_throughput                    Global Load Throughput  1151.7GB/s  1151.7GB/s  1151.7GB/s
gst_throughput                   Global Store Throughput  427.53GB/s  427.53GB/s  427.53GB/s

用nvprof 计算 kernel 运行时间：
==78622== Profiling application: ./reduceInteger
==78622== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.31%  70.411ms         6  11.735ms  11.576ms  12.037ms  [CUDA memcpy HtoD]
                    0.34%  242.69us         1  242.69us  242.69us  242.69us  reduceGmem(int*, int*, unsigned int)
                    0.32%  231.84us         6  38.639us  12.159us  65.600us  [CUDA memcpy DtoH]
                    0.31%  220.67us         1  220.67us  220.67us  220.67us  reduceSmem(int*, int*, unsigned int)
                    0.31%  220.67us         1  220.67us  220.67us  220.67us  reduceSmemDyn(int*, int*, unsigned int)
                    0.16%  114.91us         1  114.91us  114.91us  114.91us  reduceGmemUnroll(int*, int*, unsigned int)
                    0.12%  88.959us         1  88.959us  88.959us  88.959us  reduceSmemUnroll(int*, int*, unsigned int)
                    0.12%  88.928us         1  88.928us  88.928us  88.928us  reduceSmemUnrollDyn(int*, int*, unsigned int)

可以发现，用动态内存分配共享内存实现的kenrel 函数和用静态分配共享内存实现的kernel函数之间没有显著的差异。
显然，可以通过展开块来获得有效带宽的显著改进。这样做使得每个线程在运行中同时有多个内存请求，这会导致内存总线高饱和。
*/