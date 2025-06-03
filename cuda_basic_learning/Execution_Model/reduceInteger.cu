/*
避免分支分化

有时候，控制流依赖于县城索引。线程束中的条件执行可能引起分支分化，这会导致kernel 性能变差
通过重新组织数据的获取模式，可以减少或避免线程束分化。
以并行归约为例，介绍避免分支分化的基本技术

并行归约问题
假设要对一个有N个元素的整数数组进行求和。使用串行代码可以很容易实现

若有大量元素：
如何通过并行进行快速求和？
鉴于加法的结合律和交换律，数组元素可以以任何顺序求和。
并行加法计算算法：
1.将输入向量划分到更小的数据块中
2. 用一个线程计算一个数据块的部分和
3. 对每个数据块的部分和再求和得出最终结果

并行加法的一个常用方法是使用迭代成对实现。一个数据块只包含一对元素
并且一个线程对这两个元素求和产生一个局部结果。然后，这些局部结果在最初的输入向量中就地保存
这些数组作为下一次迭代求和的输入值。
因为输入值的数量在每一次迭代后减半，当输出向量的长度达到1时，最终的和就被计算出来了

根据每次迭代后的输出元素就地存储的位置，成对的并行求和实现可以进一步分为两种类型：
1. 相邻配对：元素与它们直接相邻的元素配对
2. 交错配对：根据给定的跨度配对元素
在向量中，执行满足结合律和交换律的运算，被称为归约问题。并行归约问题是这种运算的并行执行
并行归约是一种最常见的并行模式，并且是许多并行算法的一个关键运算。
在本节中，会实现多个不同的并行归约核函数，并且将测试不同的实现是如何影响kernel性能的。
*/

/*
并行归约中的分化
在相邻配对方法的的kernel 实现中，每个线程将相邻的两个元素相加产生部分和
在这个 kernel 里，有两个全局内存数组：
1个大数组用来存放整个数组，进行归约；
另1个小数组用来存放每个线程块的部分和。每个block在数组的一部分上独立地执行操作
循环中迭代一次执行一个归约步骤。归约是在就地完成的，这意味着在每一步，全局存在里的值被部分和替代
__syncthreads 语句可以保证，block 中的任一线程在进入下一次迭代之前，
在当前迭代中每个线程的所有部分和 都被保存在了全局内存中。
进入下一次迭代的所有线程都使用上一步产生的数值。在最后一个循环以后，整个线程块的和都被保存进全局内存中。
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This code implements the interleaved and neighbor-paired approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used. A
 * variety of optimizations on parallel reduction aimed at reducing divergence
 * are also demonstrated, such as unrolling.
 */

// Recursive Implementation of Interleaved Pair Approach
  /*
        因为block间的线程无法同步所以每个block的部分和和被复制回主机，并且在那进行串行求和
    */
int recursiveReduce(int *data,int const size){
    //terminate check
    if (size == 1){
        return data[0];
    }

    //renew the stride
    int const stride = size / 2;

    //inplace-reduction
    for (int i=0;i<stride;i++){
        data[i] += data[i+stride];
    }

    //call recursively
    return recursiveReduce(data,stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored(int *g_idata,int *g_odata,unsigned int n){
    /*
    在这个 kernel 里，有两个全局内存数组：
    1个大数组用来存放整个数组，进行归约；(g_idata)
    另1个小数组用来存放每个线程块的部分和。每个block在数组的一部分上独立地执行操作 (g_odata)
    循环中迭代一次执行一个归约步骤。归约是在就地完成的，这意味着在每一步，全局内存在里的值被部分和替代
    _syncthreads 语句可以保证，block 中的任一线程在进入下一次迭代之前，
    在当前迭代中每个线程的所有部分和 都被保存在了全局内存中。
    进入下一次迭代的所有线程都使用上一步产生的数值。在最后一个循环以后，整个线程块的和都被保存进全局内存中。
    */
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //boundary check
    if(idx >= n){
        return;
    }

    // in-place reduction in global memory
    //两个相邻元素间的距离被称为跨度，初始化为1。
    
    for(int stride=1; stride<blockDim.x; stride*=2){
        /*
        在每一次归约循环结束后，这个间隔会被乘以2
        在第一次循环结束后 idata(全局数据指针) 的偶数原酸将会被部分和替代
        在第二次循环结束后，idaata每4个元素将会被新产生的部分和替代
        */
        if(tid % (2*stride)==0){
            /*
            循环中迭代一次执行一个归约步骤。归约是在就地完成的，这意味着在每一步，全局内存在里的值被部分和替代
            
            
            */
            idata[tid] += idata[tid+stride];
        }
        //synchronize within block
        /*
        _syncthreads 语句可以保证，
        block 中的任一线程在进入下一次迭代之前，在当前迭代中每个线程的所有部分和都被保存在了全局内存中。
        进入下一次迭代的所有线程都使用上一步产生的数值。
        */
        __syncthreads();
    }

    // write result for this block to global mem;
    /*
        因为block间的线程无法同步所以每个block的部分和和被复制回主机，并且在那进行串行求和
    */
    if(tid == 0){
        /*
        在最后一个循环以后，整个线程块的和都被保存进全局内存中。
        */
        g_odata[blockIdx.x] = idata[0];
    }

}

/*
改善并行归约的分化
核函数 reduceNeighbored中，我们可以发现如下表达式：
if((tid % (2*stride))==0)
显然这一条件语句使得该函数只对偶数 ID 的线程为true
所以这会导致很高的线程束分化
在并行归约的第一步，只有ID为偶数的线程执行这个条件语句的主体，但是所有的线程都必须被调度
在第二次迭代中，只有1/4的线程是活跃的，但是所有的线程仍然都必须被调度
通过重新组织每个线程的数组索引来强制ID相邻的线程执行求和操作，线程束分化就能被归约了
下面的核函数中，部分和的存储位置并没有改变，但是worker thread 已经更新了：
 */

// Neighbored Pair Implementation with less warp divergence
__global__ void reduceNeighboredLess(int *g_idata,int *g_odata,unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //boundary check
    if(idx>=n){
        return;
    }

    // in-place reduction in global memory
    for(int stride=1;stride<blockDim.x;stride*=2)
    {
        //convert tid into local array index;
        //该 kernel 中下方的语句，为每个线程设置了数组访问索引
        int index = 2 * stride * tid;
        //因为跨度都乘以了2，所以下面的语句使用 block 的前半部分来执行求和操作：
        if(index < blockDim.x){
            idata[index] += idata[index+stride];
        }
        /*
        对于一个有512个线程的 block 来说，前8个线程束（32*8=256）执行第一轮归约
        剩下8个线程束均不用进行工作，因此消除了线程束分化问题
        同理，在第二轮中，前4个 warp 执行归约运算，剩下的12个warp不用执行工作
        因此，这样就彻底不存在分化了。
        然而，在最后的5轮中，当每一轮的线程总数小于 warp 的大小时，分化就会出现
        */

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid==0){
        g_odata[blockIdx.x] = idata[0];
    }
}

/*
交错配对的归约

与相邻配对方法相比，交错配对方法颠倒了元素的跨度，初始跨度是线程块的一半，
然后在每次迭代中，stride 减少一半，
在每次循环中，每个线程对两个被当前跨度隔开的元素进行求和，以产生一个部分和
交错归约的工作线程没有变化，但是每个线程在global memory中的load/store 位置是不同的
交错归约的 kernel 代码如下所示：
*/

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int *g_idata,int *g_odata,unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //convert global data pointer to the local pointer of this block;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //boundary check
    if(idx>=n){
        return;
    }

    //in-place reduction in global memory
    /*
    本 kernel 逻辑中，两个元素间的跨度被初始化为 block 大小的一半，然后在每次循环中减少一半
    */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            //在第一次迭代时，强制block的前半部分的warp执行求和操作，后半部分不进行工作
            //在第二次迭代时，仅线程块的前1/4 warp执行操作，以此类推
            //相比于reduceNeighboredLess，性能提升主要来自于 global memory 的 load/store模式导致的
            //由离散间隔到连续
            //该 kernel 与 reduceNeighboredLess 维持相同的 warp 分化
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
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

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

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
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu NeighboredLess elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);
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

    /*
    具体来说，上述三种实现方式性能如下：
    ./reduceInteger starting reduction at device 0: Tesla V100-SXM2-16GB     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.057573 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.000599 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess elapsed 0.000325 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.000283 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
Interleaved 相比于 NeighboredLess 主要性能改进体现在对于global_memory的 load/store效率的上升，这是因为读取内存由离散到连续，效率提升
后续会详细研究

观察每个warp执行的指令的数量（原始版本 Neighbore是两个优化版本的两倍多，这是因为分化后，不符合条件的指令也要被阻塞然后执行，这样一来，导致指令数量翻倍）：
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  319.875000  319.875000  319.875000

    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  124.125000  124.125000  124.125000
    
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  142.375000  142.375000  142.375000
    
观察分支效率：如之前所述，NeighboredLess 和 Interleaved本质上分支分化效率一致，均来自于stride < 32 的最后5次reduce
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                         branch_efficiency                         Branch Efficiency      77.83%      77.83%      77.83%

    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                         branch_efficiency                         Branch Efficiency      98.25%      98.25%      98.25%
    
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                         branch_efficiency                         Branch Efficiency      98.25%      98.25%      98.25%

最后观察 load 吞吐量说明 Interleaved的优化：
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  914.26GB/s  914.26GB/s  914.26GB/s
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  1732.7GB/s  1732.7GB/s  1732.7GB/s
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  518.37GB/s  518.37GB/s  518.37GB/s

首先，Neighbored 和 NeighboredLess 本质上均是在离散的index 上面操作global memory，所以其行为模式一致，但是 NeighboredLess可以取得更高的效率，因此性能更优秀
对于 InterLeaved，显然，因为连续存取，优化了加载/存储的效率，使其访存量大量下降，虽然表面上吞吐量下降， 但是其load 效率取得了显著提升：
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
   
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.02%      25.02%      25.02%
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.02%      25.02%      25.02%
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      96.15%      96.15%      96.15%

同理，在store 方面也出现了类似的趋势：
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gst_throughput                   Global Store Throughput  457.86GB/s  457.86GB/s  457.86GB/s
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gst_throughput                   Global Store Throughput  869.12GB/s  869.12GB/s  869.12GB/s
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gst_throughput                   Global Store Throughput  261.07GB/s  261.07GB/s  261.07GB/s


Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gst_efficiency            Global Memory Store Efficiency      95.52%      95.52%      95.52%

    */

}