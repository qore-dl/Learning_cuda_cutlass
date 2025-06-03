/*
 展开循环
 循环展开是一个尝试通过减少分支出现的频率和循环维护指令来优化循环的技术。
 在循环展开中，循环主体在代码中要多次被编写，而不是只编写一次循环主体在使用另一个循环来反复执行的。
 任何的封闭循环可以将它的迭代次数减少或完全删除。
 循环体的复制数量被称为循环展开因子
 迭代次数就变为了原始循环迭代次数除以循环展开因子。
 在顺序数组中，当循环的迭代次数在循环执行之前就已经知道时，循环展开是最有效提升性能的方法。
 考虑以下代码：
 for(int i;i<100;i++){
    a[i] = b[i] + c[i];
 }
 如果重复操作一次循环体，迭代次数能减少到原始循环的一半：
 for(int i;i<100;i+=2){
    a[i] = b[i] + c[i];
    a[i+1] = b[i+1] + c[i+1];
 }

从高级语言层面上来看，循环展开使得性能提高的原因可能不是那么显而易见的。
这种提升来自于编译器执行循环展开时低级指令的改进和优化。
例如，在前面循环展开的例子中，每个语句的读和写都是独立的，所以CPU可以同时发出内存操作。

在CUDA中，循环展开的意义非常重大。我们的目标仍然是相同的：通过减少指令消耗和增加更多的独立调度指令来提高性能。
因此，更多的并行操作被添加到流水线上，以产生更高的指令和内存带宽/吞吐量
这为线程束调度器提供更多符合条件的线程束，它们可以帮助隐藏指令或内存延迟。
*/



/*

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
       
        */
        if(tid % (2*stride)==0){
            /*   
            */
            idata[tid] += idata[tid+stride];
        }
        //synchronize within block
        /*
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


/*
展开的归约
在reduceInterleaved kernel 函数中，每个线程块只处理一部分数据，这些数据可以被认为是一个数据块。
如果用一个block 手动展开两个数据块的处理，可以展开循环提升性能。
以下的 kernel 是 reduceInterleaved kernel 函数的修正版，每个block汇总了来自两个数据块的数据。
这是一个循环分区的例子，每个线程作用于多个数据块，并处理每个数据块的一个元素。
*/

__global__ void reduceUnrolling2(int *g_idata,int *g_odata,unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;

    /*
    全局数组索引被相应地调整，因为只需要一半的block 块来处理相同的数据集。
    请注意，这也意味着对于相同大小的数据集，向设备展示的wrap 和 block 级别的并行性更低
    */
    unsigned int idx = blockIdx.x * blockDim.x*2 + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x*2;

    //unrolling 2 data blocks
    /*
    在kernel 函数开头添加了下述的语句。
    在这里，每个线程都添加一个来自于相邻数据块的元素。
    从概念上讲，可以把它作为归约循环的一个迭代，
    此循环可在数据块间归约：
    */
    if(idx + blockDim.x<n){
        g_idata[idx] += g_idata[idx+blockDim.x]; 
    }
    __syncthreads();

    //in-place reduction in global memory
    for(int stride=blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }

}

__global__ void reduceUnrolling4(int *g_idata,int *g_odata,unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;

    /*
    全局数组索引被相应地调整，因为只需要一半的block 块来处理相同的数据集。
    请注意，这也意味着对于相同大小的数据集，向设备展示的wrap 和 block 级别的并行性更低
    */
    unsigned int idx = blockIdx.x * blockDim.x*4 + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x*4;

    //unrolling 2 data blocks
    /*
    在kernel 函数开头添加了下述的语句。
    在这里，每个线程都添加一个来自于相邻数据块的元素。
    从概念上讲，可以把它作为归约循环的一个迭代，
    此循环可在数据块间归约：
    */
    if((idx + 3*blockDim.x) < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx+blockDim.x];
        int a3 = g_idata[idx+blockDim.x*2];
        int a4 = g_idata[idx+blockDim.x*3];
        g_idata[idx] = a1 + a2 + a3 + a4;
       
    }
    __syncthreads();

    //in-place reduction in global memory
    for(int stride=blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }

}

__global__ void reduceUnrolling8(int *g_idata,int *g_odata,unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;

    /*
    全局数组索引被相应地调整，因为只需要一半的block 块来处理相同的数据集。
    请注意，这也意味着对于相同大小的数据集，向设备展示的wrap 和 block 级别的并行性更低
    */
    unsigned int idx = blockIdx.x * blockDim.x*8 + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x*8;

    //unrolling 2 data blocks
    /*
    在kernel 函数开头添加了下述的语句。
    在这里，每个线程都添加一个来自于相邻数据块的元素。
    从概念上讲，可以把它作为归约循环的一个迭代，
    此循环可在数据块间归约：
    */
    if((idx + 7*blockDim.x) < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx+blockDim.x];
        int a3 = g_idata[idx+blockDim.x*2];
        int a4 = g_idata[idx+blockDim.x*3];
        int a5 = g_idata[idx+blockDim.x*4];
        int a6 = g_idata[idx+blockDim.x*5];
        int a7 = g_idata[idx+blockDim.x*6];
        int a8 = g_idata[idx+blockDim.x*7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
       
    }
    __syncthreads();

    //in-place reduction in global memory
    for(int stride=blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
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

    // kernel 4: reduceUnrolling2
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling2<<<grid.x/2,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    //因为每个 block 处理两个数据块，我们需要调整内核的执行配置，将网格大小减小至一半：
    cudaMemcpy(h_odata,d_odata,grid.x/2*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/2;i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu Unrolling2 elapsed %f sec gpu_sum: %d <<grid %d block %d>>\n",iElaps,gpu_sum,grid.x/2,block.x);

    // kernel 5: reduceUnrolling4
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling4<<<grid.x/4,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    //因为每个 block 处理两个数据块，我们需要调整内核的执行配置，将网格大小减小至一半：
    cudaMemcpy(h_odata,d_odata,grid.x/4*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/4;i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu Unrolling4 elapsed %f sec gpu_sum: %d <<grid %d block %d>>\n",iElaps,gpu_sum,grid.x/4,block.x);

    // kernel 6: reduceUnrolling8
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling8<<<grid.x/8,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    //因为每个 block 处理两个数据块，我们需要调整内核的执行配置，将网格大小减小至一半：
    cudaMemcpy(h_odata,d_odata,grid.x/8*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/8;i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu Unrolling8 elapsed %f sec gpu_sum: %d <<grid %d block %d>>\n",iElaps,gpu_sum,grid.x/8,block.x);

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
    如预想的一样，在一个线程中有更多的独立内存 load/store 操作会产生更好的性能（线程内部流水线式并发执行，且线程间可以完美并行），
    因为内存延迟可以被更好地隐藏起来
    可以使用设备内存读取吞吐量指标来确这是性能提升的原因：

    整体性能：
    ./Unrolling starting reduction at device 0: Tesla V100-SXM2-16GB     with array size 16777216  grid 32768 block 512
cpu reduce      elapsed 0.057168 sec cpu_sum: 2139353471
gpu Neighbored  elapsed 0.000600 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu NeighboredLess elapsed 0.000324 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Interleaved elapsed 0.000284 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
gpu Unrolling2 elapsed 0.000181 sec gpu_sum: 2139353471 <<grid 16384 block 512>>
gpu Unrolling4 elapsed 0.000146 sec gpu_sum: 2139353471 <<grid 8192 block 512>>
gpu Unrolling8 elapsed 0.000138 sec gpu_sum: 2139353471 <<grid 4096 block 512>>

Unrolling2/4/8 相比于 Neighbored 和 Interleaved 而言，显著提升

主要体现在片内内存读取的吞吐提升，同时这一独立加载连续空间，相比于 Interleaved，进一步提升了global memory load 的吞吐量与效率：
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceUnrolling8(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  587.61GB/s  587.61GB/s  587.61GB/s
    Kernel: reduceUnrolling4(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  553.47GB/s  553.47GB/s  553.47GB/s
    Kernel: reduceUnrolling2(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  422.40GB/s  422.40GB/s  422.40GB/s

    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  249.37GB/s  249.37GB/s  249.37GB/s
    
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  114.62GB/s  114.62GB/s  114.62GB/s
    
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  217.33GB/s  217.33GB/s  217.33GB/s


Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceUnrolling8(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  740.00GB/s  740.00GB/s  740.00GB/s
    Kernel: reduceUnrolling4(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  836.72GB/s  836.72GB/s  836.72GB/s
    Kernel: reduceUnrolling2(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  860.54GB/s  860.54GB/s  860.54GB/s
    
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  518.24GB/s  518.24GB/s  518.24GB/s
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  914.69GB/s  914.69GB/s  914.69GB/s
    
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  1733.3GB/s  1733.3GB/s  1733.3GB/s

注意，Unrolling 2/4/8 相比于 Interleaved，性能吞吐量上升，但是因为片上内存吞吐量，利用效率的提升，在Unrolling 2/4/8，随着展开因子增大，对全局内存吞吐量下降，但是gld_efficiency 上升：

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceUnrolling8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.21%      99.21%      99.21%
    Kernel: reduceUnrolling4(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      98.68%      98.68%      98.68%
    Kernel: reduceUnrolling2(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      98.04%      98.04%      98.04%
          
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      96.15%      96.15%      96.15%

    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.02%      25.02%      25.02%
    
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.02%      25.02%      25.02%
    */

}