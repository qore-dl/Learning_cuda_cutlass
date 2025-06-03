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

/*
展开线程的归约
__syncthreads是用于 block 内同步的，
在上述的归约 kernel 函数中，它用来确保在线程进入下一轮之前，每一轮中所有线程已经将局部结果写入全局内存中了
然而，要细想一下 只剩 32个或更少的线程（即仅剩一个线程束执行工作）的情况。因为线程束的执行是SIMT(单指令多线程)的，
每条指令之后有隐式的线程束内同步过程，因此归约循环的最后6个迭代实际上是可以进行展开的，如下面的 kernel 所示：
 */
__global__ void reduceUnrollingWarps8 (int *g_idata, int *g_odata, unsigned int n){
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
    for(int stride=blockDim.x/2; stride > 32; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        //synchronize within threadblock
        __syncthreads();
    }

    /*
    因为线程束的执行是SIMT(单指令多线程)的，
    每条指令之后有隐式的线程束内同步过程，因此归约循环的最后6个迭代实际上是可以进行展开的如下所示：
    */
    //unrolling warp
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
        // g_odata[blockIdx.x] = idata[0];
    }
    /*
    这个线程束的展开避免了执行循环控制和线程在block内夸warp同步的逻辑，warp内部的SIMT模式具有隐式同步逻辑
    注意，变量 vmem 是和volatile 修饰符一起被声明的，它告诉编译器，每次赋值时必须将vmem[tid]的值存回到全局内存中
    如果省略了volatile 修饰符，这段代码将不能正常工作，因为编译器或缓存可能会对global 或 shared memory 优化读写。
    如果位于 global 或 shared memory 中的变量有 volatile 修饰符，
    编译器会假定其值可以被其他线程在任何时间修改或使用，
    因此，任何参考 volatile 修饰符的变量会强制直接读或写内存，而不是简单地读写缓存或寄存器

    */

    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

/*
完全展开的归约

如果编译时，已知一个循环中的迭代次数，就可以把循环完全展开，因为在Fermi 或 Kepler 架构中，每个 block 的最大线程数是1024
并且在这些归约 kernel 函数中循环迭代次数是基于一个block 维度的，所以完全展开这些归约循环是有可能的：
*/

__global__ void reduceCompleteUnrollWarps8(int *g_idata,int *g_odata,unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x*8 + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    //unrolling 8
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

    //in-place reduction and complete unroll
    if(blockDim.x>=1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if(blockDim.x>=512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if(blockDim.x>=256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if(blockDim.x>=128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    //unrolling warp
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
        // g_odata[blockIdx.x] = idata[0];
    }
    /*
    这个线程束的展开避免了执行循环控制和线程在block内夸warp同步的逻辑，warp内部的SIMT模式具有隐式同步逻辑
    注意，变量 vmem 是和volatile 修饰符一起被声明的，它告诉编译器，每次赋值时必须将vmem[tid]的值存回到全局内存中
    如果省略了volatile 修饰符，这段代码将不能正常工作，因为编译器或缓存可能会对global 或 shared memory 优化读写。
    如果位于 global 或 shared memory 中的变量有 volatile 修饰符，
    编译器会假定其值可以被其他线程在任何时间修改或使用，
    因此，任何参考 volatile 修饰符的变量会强制直接读或写内存，而不是简单地读写缓存或寄存器

    */

    //write result for this block to global mem
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

/*
模版函数的归约
虽然可以手动展开循环，但是使用模版函数有助于进一步减少分支消耗。
在设备函数上 CUDA 支持模版参数。
如下所示，可以指定 block 的大小作为模版函数的参数：
*/

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n){
    //set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    //unrolling 8
    if((idx + 7*blockDim.x) < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    //in-place reduction and complete unroll
    if(iBlockSize>=1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if(iBlockSize>=512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if(iBlockSize>=256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if(iBlockSize>=128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    //unrolling warp
    if(tid < 32){
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    //write results for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];

}
/*
相比于 reduceCompleteUnrollWarps8，唯一的区别就是使用了模版参数替换了block的大小。
模版参数化的目的：
检查参数化块大小的if 语句将在编译时被评估，如果这一条件为false，那么编译时将它删除，使得内循环效率更高
例如，在线程块大小为256的情况下调用这个核函数，则 iBlockSize>=1024 && tid < 512 这一条件永远为false
因此，编译器会自动将其从执行kernel中移除

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

    // kernel 7: reduceUnrollingWarps8
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    //在这个实现中，每个线程处理8个数据块，调用这个 kernel 的同时，它的grid 尺寸减小到1/8：
    reduceUnrollingWarps8<<<grid.x/8,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    //因为每个 block 处理两个数据块，我们需要调整内核的执行配置，将网格大小减小至一半：
    cudaMemcpy(h_odata,d_odata,grid.x/8*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/8;i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu UnrollingWarps8 elapsed %f sec gpu_sum: %d <<grid %d block %d>>\n",iElaps,gpu_sum,grid.x/8,block.x);

    // kernel 8: reduceCompleteUnrollWarps8
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    //在这个实现中，每个线程处理8个数据块，调用这个 kernel 的同时，它的grid 尺寸减小到1/8：
    reduceCompleteUnrollWarps8<<<grid.x/8,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    //因为每个 block 处理两个数据块，我们需要调整内核的执行配置，将网格大小减小至一半：
    cudaMemcpy(h_odata,d_odata,grid.x/8*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/8;i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu CompleteUnrollingWarps8 elapsed %f sec gpu_sum: %d <<grid %d block %d>>\n",iElaps,gpu_sum,grid.x/8,block.x);

    //kernel 9: 模板化的reduceCompleteUnroll
    /*
    该模版化的核函数一定要在switch-case 结构中被调用。
    这允许编译器为特定的block大小自动优化代码，但这也意味着它只能对特定 block 大小启动的reduceCompleteUnroll函数有效：
    */

    // unsigned int blocksize
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();

    switch (blocksize){
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x/8,block>>>(d_idata,d_odata,size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x/8,block>>>(d_idata,d_odata,size);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid.x/8,block>>>(d_idata,d_odata,size);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid.x/8,block>>>(d_idata,d_odata,size);
            break;
        case 64:
            reduceCompleteUnroll<64><<<grid.x/8,block>>>(d_idata,d_odata,size);
            break;
    }

    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    //因为每个 block 处理两个数据块，我们需要调整内核的执行配置，将网格大小减小至一半：
    cudaMemcpy(h_odata,d_odata,grid.x/8*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/8;i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu CompleteUnrollingWarps elapsed %f sec gpu_sum: %d <<grid %d block %d>>\n",iElaps,gpu_sum,grid.x/8,block.x);

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
    ./Unrollingwarps starting reduction at device 0: Tesla V100-SXM2-16GB     with array size 16777216  grid 32768 block 512
    cpu reduce      elapsed 0.057381 sec cpu_sum: 2139353471
    gpu Interleaved elapsed 0.000304 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
    gpu Unrolling8 elapsed 0.000150 sec gpu_sum: 2139353471 <<grid 4096 block 512>>
    gpu UnrollingWarps8 elapsed 0.000142 sec gpu_sum: 2139353471 <<grid 4096 block 512>>

    相比于原来的Unrolling8，UnrollingWarps8的整体速度略有提升，相比于Interleaved大幅提升

    其来源于对于block 内 多余的 跨 warp 同步的减少，使用stall_sync指标来来证实：
    （由于 __syncthreads 的同步，更少的线程束发生阻塞）
    ==121052== Profiling application: ./Unrollingwarps
    ==121052== Profiling result:
    ==121052== Metric result:
    
    Invocations                               Metric Name                        Metric Description         Min         Max         Avg
    Device "Tesla V100-SXM2-16GB (0)"
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      44.51%      44.51%      44.51%
    Kernel: reduceUnrolling8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      16.55%      16.55%      16.55%
    Kernel: reduceUnrollingWarps8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      10.05%      10.05%      10.05%

    相比于 Unrolling8而言，我们发现，通过展开最后的warp，因为同步导致的stall几乎减半了，
    这表明了合理使用 __syncthreads 能减少新的kernel 函数中的阻塞

      完全展开后，端到端性能略有上升：
    ./Unrollingwarps starting reduction at device 0: Tesla V100-SXM2-16GB     with array size 16777216  grid 32768 block 512
    cpu reduce      elapsed 0.057766 sec cpu_sum: 2139353471
    gpu Interleaved elapsed 0.000311 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
    gpu Unrolling8 elapsed 0.000156 sec gpu_sum: 2139353471 <<grid 4096 block 512>>
    gpu UnrollingWarps8 elapsed 0.000149 sec gpu_sum: 2139353471 <<grid 4096 block 512>>
    gpu CompleteUnrollingWarps8 elapsed 0.000148 sec gpu_sum: 2139353471 <<grid 4096 block 512>>

    模板化后end-to-end 略有上升：
    ./Unrollingwarps starting reduction at device 0: Tesla V100-SXM2-16GB     with array size 16777216  grid 32768 block 512
    cpu reduce      elapsed 0.057452 sec cpu_sum: 2139353471
    交错 gpu Interleaved elapsed 0.000307 sec gpu_sum: 2139353471 <<<grid 32768 block 512>>>
    展开8块 gpu Unrolling8 elapsed 0.000144 sec gpu_sum: 2139353471 <<grid 4096 block 512>>
    展开8块+最后的线程束 gpu UnrollingWarps8 elapsed 0.000154 sec gpu_sum: 2139353471 <<grid 4096 block 512>>
    展开8块+循环+最后的线程束 gpu CompleteUnrollingWarps8 elapsed 0.000146 sec gpu_sum: 2139353471 <<grid 4096 block 512>>
    模板化内核 gpu CompleteUnrollingWarps elapsed 0.000143 sec gpu_sum: 2139353471 <<grid 4096 block 512>>

    同时stall sync下降：
    Invocations                               Metric Name                        Metric Description         Min         Max         Avg
    Device "Tesla V100-SXM2-16GB (0)"
    交错 Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      44.51%      44.51%      44.51%
    展开8块 Kernel: reduceUnrolling8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      16.24%      16.24%      16.24%
    展开8块+最后的线程束 Kernel: reduceUnrollingWarps8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)       9.63%       9.63%       9.63%

    展开8块+循环+最后的线程束 Kernel: reduceCompleteUnrollWarps8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      10.58%      10.58%      10.58%
    
    模板化内核 Kernel: void reduceCompleteUnroll<unsigned int=512>(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)       9.44%       9.44%       9.44%
    
    
    同时，我们可以发现，最大的性能增益来自于reduceUnrolling8 kernel 函数
    在这个函数中，每个线程在进行循环归约之前处理了8个数据块
    即，有了8个独立的内存访问，可以更好地让内存带宽饱和及隐藏加载/存储延迟
    可以使用以下命令检测 global memory load/store 效率指标：

    nvprof --metrics gld_efficiency,gst_efficiency ./Unrollingwarps

    Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    交错 Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      96.15%      96.15%      96.15%
          1                            gst_efficiency            Global Memory Store Efficiency      95.52%      95.52%      95.52%
    
    展开8块（边际增益最大） Kernel: reduceUnrolling8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.21%      99.21%      99.21%
          1                            gst_efficiency            Global Memory Store Efficiency      97.71%      97.71%      97.71%
    
    展开8块+最后的线程束 Kernel: reduceCompleteUnrollWarps8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.43%      99.43%      99.43%
          1                            gst_efficiency            Global Memory Store Efficiency      99.40%      99.40%      99.40%
   
    展开8块+循环+最后的线程束 Kernel: reduceUnrollingWarps8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.43%      99.43%      99.43%
          1                            gst_efficiency            Global Memory Store Efficiency      99.40%      99.40%      99.40%

    模板化内核 Kernel: void reduceCompleteUnroll<unsigned int=512>(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.43%      99.43%      99.43%
          1                            gst_efficiency            Global Memory Store Efficiency      99.40%      99.40%      99.40%
    
    */

  

}