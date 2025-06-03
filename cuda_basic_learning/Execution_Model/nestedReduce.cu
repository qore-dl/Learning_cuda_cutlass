/*
动态并行

到目前为止，所有的kernel 函数都是从主机线程中被调用的。GPU 的工作负载完全在 CPU 的控制下。
CUDA 的动态并行允许在GPU端直接创建和同步新的GPU内核。
在一个kernel 函数中，在任意点动态增加 GPU 应用程序的并行性，是一个令人兴奋的新功能

到目前为止，我们需要把算法设计为单独的、大规模数据并行的内存启动。
动态并行提供了一个更有层次结构的方法，在这个方法中，并发性可以在一个GPU内核的多个level中表现出来
使用动态并行可以让递归算法更加清晰易懂，也更容易理解

有了动态并行，可以推迟到运行时决定需要在 GPU 上创建多少个 block 和 grid，
可以动态地利用 GPU 硬件调度器和加载平衡器，并进行调整以适应数据驱动或工作负载
在GPU端直接创建工作的能力可以减少在主机和设备之间传输执行控制和数据的需求，因为在设备上执行的线程可以在运行时决定启动配置

在本节中，将通过使用动态并行实现递归归约kernel 函数的例子，对如何利用动态并行做一个基本的理解
*/

/*
1. 嵌套执行
通过动态并行，我们已经熟悉了内核执行的概念（grid,block,启动配置等），也可以直接在GPU上进行kernel调用
相同的kernel 调用语法被用于在一个kernel 内部启动另一个新的kernel 函数

在动态并行中，kenrel 执行分为两种类型：父母和孩子（parent and child）
父线程、父block 或 父 grid启动一个新的grid，即子grid。
子线程、子 block 或 子 grid 被parent 启动。
子 grid 必须在父线程、父 block 或 父 grid 之前完成。
只有在所有的子 grid 都完成之后，parent 才会完成。

父grid 和子 grid 的适用范围：
host 线程配置和启动 父 grid，父grid配置和启动子 grid。
子grid的调用和完成必须进行适当地嵌套，这意味着在线程创建的所有子grid都完成之后，父grid才会完成
如果调用的thread没有显式地同步启动子网格，那么cuda runtime会保证 parent 和 child 之间的隐式同步。
显式同步例子：父 thread 可以设置栅栏，从而可以与其子 grid 显式地同步

device 线程中的 grid 启动，在 block 内是可见的。这意味着，线程可能与由该线程启动的或由相同block 中其他线程启动的子 grid 同步
在 block 中，只有当所有线程创建的所有子 grid 完成之后，block 的执行才会完成。
如果 block 中所有线程在所有的子 grid 完成之前退出，那么在那些子 grid 上隐式同步会被触发。

当parent 启动一个子 grid，父 block 与 child 显式同步之后，child才能开始执行。
父 grid 和 子 grid 共享相同的global 和 constant memory 存储
但是父 grid 和 子 grid 之间有不同的 local memory 和 shared memory
有了 child 和 parent 之间的弱一致性做保证，父 grid 和 子 grid 可以对全局内存并发load/store
有两个时刻，子 grid 和它的父 thread见到的内存完全相同：
1. 子 grid 开始时：当父thread优于子grid 调用时，所有的全局内存操作要保证对子 grid 是可见的。
2. 子 grid 结束时：当parent在子 grid 完成时进行同步操作后，子 grid 所有的内存操作应保证对parent 是可见的。

shared memory 和 local memory 分别对于block 或 thread 而言是私有的，同时，在parent 和 child 之间不是可见或一致的。
local memory 对于线程来说是私有存储，并且对该线程外部不可见
当启动一个 子 grid时，传递一个指向local memory的指针 作为参数是无效的。



*/

/*
嵌套归约

归约可以被表示为一个递归函数，在前文已经对其分支分化的避免进行了大量的研究。
在CUDA中可以使用动态并行，确保 CUDA 里的递归函数的实现像在 C 语言中一样简单

下面列出了带有动态并行的递归归约的 kernel 代码。
这个 kernel 函数中，原始的grid包含许多 block，但所有嵌套的子 grid 中有且只有一个由其父 grid 的线程0调用的block

kernel 函数的第一步是将global memory 地址 g_idata 转换为每个block的本地地址
接下来，如果满足停止条件（该block为嵌套执行树上的leaf节点），结果就被拷贝回global memory
并且控制立刻返回到parent kernel 中。
如果它不是一个leaf child kernel，就需要计算本地归约的规模，一半的线程执行就地归约。
在就地归约完成后，同步block以保证所有部分和的计算。
紧接着，线程0产生一个只有1个block和1个当前父block一半线程数量的子 grid。
在子 grid 被调用后，所有子 grid 都会设置一个障碍点。
因为在每个block里，一个线程只产生一个字 grid，所以这个障碍点只会同步一个 子 grid。
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"

__global__ void gpuRecursiveReduce(int *g_idata,int *g_odata,unsigned int isize){
    // set thread ID
    unsigned int tid = threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    //stop condition
    if(isize == 2 && tid == 0){
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    //nested invocation
    int istride = isize >> 1;
    if(istride > 1 && tid < istride){
        // in place reduction
        idata[tid] += idata[tid+istride];
    }

    //sync at block level
    __syncthreads();

    //nested invocation to generate child grids
    if(tid==0){
        gpuRecursiveReduce<<<1,istride>>>(idata,odata,istride);

        // sync all child grids launched in this block;
        cudaDeviceSynchronize();
    }

    //sync at block level again
    __syncthreads();

}

// Recursive Implementation of Interleaved Pair Approach
int cpuRecursiveReduce(int *data, int const size)
{
    // stop condition
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return cpuRecursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/*
当一个 子 grid 被调用后，它看到的内存和父线程是一样的。
因为每一个子线程只需要父线程的数值来指导部分归约，
所以在每个子grid启动前执行block内部的同步是没有必要的(即这里有一个和父block隐式的同步操作)，同理在子grid结束前也有一个隐式的同步操作。去除所有同步操作会产生如下的kernel函数
*/
__global__ void gpuRecursiveReduceNosync(int *g_idata,int *g_odata,unsigned int isize){
    //set thread ID
    unsigned int tid = threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    //stop condition
    if(isize == 2 && tid == 0){
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    //nested invoke
    int istride = isize>>1;
    if(istride > 1 && tid < istride){
        idata[tid] += idata[tid + istride];

    }
    if(tid == 0){
        gpuRecursiveReduceNosync<<<1,istride>>>(idata,odata,istride);
    }
}

/*
相较于同步的动态执行，时间仅为1/3，但是相比于相邻配对的kernel而言，性能仍然很差
需要考虑如何减少由大量的子grid 启动引发的消耗。在当前的实现中，每个block产生一个子 grid，并且引起了大量的调用
因此可以考虑：当创建的子grid数量减少，每个grid内部block 增多的思路，在保证相同数量的并行性和弱一致性前提下优化性能
*/

__global__ void gpuRecursiveReduce2(int *g_idata,int *g_odata,int iStride,int const iDim){
    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * iDim;

    // stop condition
    if (iStride == 1 && threadIdx.x == 0){
        g_odata[blockIdx.x] = idata[0]+idata[1];
        return;
    }

    // in place reduction
    idata[threadIdx.x] += idata[threadIdx.x + iStride];

    //nested invocation to generate child grids
    if(threadIdx.x == 0 && blockIdx.x == 0){
        gpuRecursiveReduce2 <<<gridDim.x,iStride/2>>>(g_idata,g_odata,iStride/2,iDim);
    }
}

int main(int argc,char **argv){
    //set up device
    int dev = 0;
    int gpu_sum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    bool bResult = false;

     // set up execution configuration
    int nblock  = 2048;
    int nthread = 512;   // initial block size

    if(argc > 1)
    {
        nblock = atoi(argv[1]);   // block size from command line argument
    }

    if(argc > 2)
    {
        nthread = atoi(argv[2]);   // block size from command line argument
    }

    int size = nblock * nthread; // total number of elements to reduceNeighbored

    dim3 block (nthread,1);
    dim3 grid((size+block.x-1)/block.x,1);

    printf("array %d grid %d block %d\n", size, grid.x, block.x);

    // allocate host memory

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)( rand() & 0xFF );
        h_idata[i] = 1;
    }

    memcpy (tmp, h_idata, bytes);

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;

    CHECK(cudaMalloc((void **) &d_idata,bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    double iStart,iElaps;

    // cpu recursive reduction
    iStart = seconds();
    int cpu_sum = cpuRecursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce\t\telapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // gpu reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored\t\telapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // gpu nested reduce kernel
    CHECK(cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduce<<<grid,block>>>(d_idata,d_odata,block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    CHECK(cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduceNosync<<<grid,block>>>(d_idata,d_odata,block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested nosync\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    CHECK(cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduce2<<<grid,block.x/2>>>(d_idata,d_odata,block.x/2,block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested2\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

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
cpu reduce              elapsed 0.003577 sec cpu_sum: 1048576
gpu Neighbored          elapsed 0.000156 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu nested              elapsed 0.034065 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>

使用嵌套动态并行的算法远远慢于相邻配对算法的kernel 实现
这是因为：最初有2048个 block。因为每个block执行8次递归（512->2)，
所以总共创建了16,384个子block，用于同步 block 内部的 __syncthreads函数也被调用了16,384次
如此大量的kernel 调用与同步很可能是造成内核效率低的主要原因

cpu reduce              elapsed 0.003636 sec cpu_sum: 1048576
gpu Neighbored          elapsed 0.000163 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu nested              elapsed 0.034037 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu nested nosync               elapsed 0.012578 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>

相较于同步的动态执行，时间仅为1/3，但是相比于相邻配对的kernel而言，性能仍然很差
需要考虑如何减少由大量的子grid 启动引发的消耗。在当前的实现中，每个block产生一个子 grid，并且引起了大量的调用
因此可以考虑：当创建的子grid数量减少，每个grid内部block 增多的思路，在保证相同数量的并行性和弱一致性前提下优化性能

./nestedReduce starting reduction at device 0: Tesla V100-SXM2-16GB array 1048576 grid 2048 block 512
cpu reduce              elapsed 0.003569 sec cpu_sum: 1048576
gpu Neighbored          elapsed 0.000154 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu nested              elapsed 0.033154 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu nested nosync               elapsed 0.012516 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>
gpu nested2             elapsed 0.000738 sec gpu_sum: 1048576 <<<grid 2048 block 512>>>

性能显著提升，这是因为调用了较少的子 grid
可以使用 nvprof 来验证性能提高的原因：
nvprof ./nestedReduce


nested 和 nested nosync 调用了16,384个子grid。gpuRecursiveReduce2 kernel 中的8层嵌套并行只创建了8个子grid：

递归归约的例子说明了动态并行，对于一个给定的算法，通过使用不同的动态并行技术，可以由多种可能得实现方式
避免大量嵌套调用有助于减少消耗并提升性能
同步对性能与正确性都至关重要，但减少block内部的同步次数可能会使嵌套kernel 效率更高
因为在每一个嵌套层上，device runtime system 都需要保留额外的内存（global,constant),
所以内核嵌套的最大数量可能是受限制的
这种限制的程度依赖于kernel，也可能会限制任何使用动态并行应用程序的扩展，性能以及其他的性能

*/

/*
总结
本章从硬件的角度分析了kernel 执行。在 GPU device 上，CUDA 执行模型有两个最显著的特性：
1. 使用SIMT的方式在线程束中执行线程
2. 在block 与 thread 中分配了硬件资源

这些执行模型的特征使得我们在提高并行性和性能时，能控制应用程序是如何让指令和内存带宽饱和的
不同计算能力的 GPU device 有不同的硬件限制，因此，grid 和 block 的启发式算法在为不同的平台优化 kernel性能方面发挥了重要的作用

动态并行使得设备能够直接创建新的工作。它确保我们可以用一种更自然和更易于理解的方式来表达递归或依赖数据并行的方法
为实现一个有效的嵌套kernel，必须注意device runtime 的使用，其包括 子 grid 启动策略、父子同步和嵌套层的深度等

本章也介绍了使用命令行分析工nprovf 详细分析kernel 性能的方法
因为一个单纯的kernel 实现可能不会产生很好的性能，所以配置文件驱动的方法在 CUDA 编程中尤其重要
性能分析对kernel 行为提供了详细的分析，并能找到产生最佳性能的主要因素。
*/