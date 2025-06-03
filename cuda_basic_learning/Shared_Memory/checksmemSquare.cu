#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>



/*
5.2 shared memory 的数据布局
为了全面了解如何有效地使用共享内存，本节将使用共享内存研究几个简单的例子,其中包括下列主题：
1.方阵与矩阵数组
2.行主序和列主序访问
3.静态与动态共享内存的声明
4.文件范围与内核范围的共享内存
5.内存填充与无内存填充

当使用共享内存设计kernel函数时，重点应该放在以下两个概念上：
1. 跨内存存储体映射数据元素
2. 从线程索引到共享内存偏移的映射

当这些概念了然于心时，就可以设计一个高效的kernel函数了，它可以避免存储体冲突，并充分利用共享内存的优势
*/

/*
5.2.1 方形共享内存
使用共享内存可以直接缓存具有方形维度的全局数据。方形矩阵的简单维度可以很容易从二维线程索引中计算出一维内存偏移。
图 5-12 显示了一个共享内存块，它在每个维度有32个元素，按行主序进行存储。上部的图显示了一维数据的实际排列，下部的图
显式了带有4字节数据元素和存储体映射的二维共享内存逻辑视图。

使用下面的语句静态声明一个二维共享内存变量：
__shared__ int tile[BDIMX][BDIMY];

字节地址：          0 4 8 12 16 20 24 28 ......  4088 4092
4字节索引：         0 1 2 3  4  5  6  7  ......  1022 1023

存储体(Bank)索引    0    1    2    3   ...  28   29   30   31
Row0    :          0    1    2    3   ...  28   29   30   31
Row1    :         32   33   34   35   ...  60   61   62   63
Row2    :         64   65   66   67   ...  92   93   94   95
Row3    :         96   97   98   99   ...  124  125  126  127
......                                           
Row30   :        960  961  962  963   ...  988  989  990  991
Row31   :        992  993  994  995   ... 1020 1021 1022 1023
                Col0 Col2 Col3 Col4   ... Co28 Co29 Co30 Co31

因为这个共享内存块是方形的，所以可以选择一个二维线程块访问它，在 x 或者 y 维度上通过相邻线程访问相邻元素：
tile[threadId.y][threadId.x]
tile[threadId.x][threadId.y]

在这些访问方法中哪个有可能表现得更好？这就需要注意线程与共享内存存储体的映射关系。
回想一下，在同一个wrap中若有访问独立存储体的线程，则它是最优的。相同线程束中的线程可由连续的threadId.x的值来确定。
属于不同存储体的共享内存元素也可以通过字偏移进行连续存储。因此，最好是有访问共享内存连续位置的线程，且该线程带有连续的threadIdx.x的值。
由此，可以得出结论，第一存取模式（块 [threadIdx.y][threadIdx.y])将比第二存取模式(块 [threadIdx.x][threadIdx.y]) 呈现出更好的性能和更少的存储体冲突，
因为邻近线程在最内层数组维度上访问相邻的阵列单元。
*/

/*
5.2.1.1 行主序访问和列主序访问
考虑一个例子，在例子中网格有一个二维线程块，块中每个维度包含32个可用的线程。可以使用下面的宏来定义块维度
*/
#define BDIMX 4
#define BDIMY 4
#define IPAD  1

/*
还可以使用下面的宏来定义kernel函数的执行配置：
dim3 block (BDIMX,BDIMY);
dim3 grid(1,1)
kernel 函数有两个简单操作：
1. 将全局线程索引按行主序写入到一个二维共享内存数组中
2. 将共享内存中按行主序读取这些值并将它们存储到全局内存中
*/

__global__ void setRowReadRow(int *out){
    //static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    //mapping from thread index to global memory index;
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //shared memory store operation:
    tile[threadIdx.y][threadIdx.x] = idx;

    //wait for all threads to complete
    __syncthreads();

    //shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];

    /*
    到目前为止，内核中有3个内存操作：
    共享内存的存储操作
    共享内存的加载操作
    全局内存的存储操作
    因为相同线程束的线程由连续的threadIdx.x值，并且可以使用threadIdx.x 索引共享内存数据tile的最内层维度，所以 kernel 函数无存储体冲突。
    */

}

/*
另一方面，如果在数据分配给共享内存块时交换threadIdx.y 和 threadIdx.x，线程束的内存将会按列主序访问。
    每个共享内存的加载和存储将导致Fermi装置中有32路存储体冲突，导致 Kepler 装置中有16路存储体冲突。
*/

__global__ void setColReadCol(int *out){
    //static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    //mapping from thread index to global memory index;
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    //wait for all threads to complete
    __syncthreads();

    //shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

/*
5.2.1.2 按行主序写和按列主序读
下面的kernel 函数实现了共享内存按行写入和按列读取。
按行主序写入共享内存是将线程索引的最内层维度作为二维共享内存块的列索引实现的（等同于最后一个例子）：
tile[threadIdx.y][threadIdx.x] = idx;
按列主序在共享内存块中给全局内存赋值，这是在引用共享内存时交换两个线程索引实现的：
out[idx] = tile[threadIdx.x][threadIdx.y];
kernel 代码如下：
*/

__global__ void setRowReadCol(int *out){
    //static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    //mapping from thread index to global memory index;
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    //wait for all threads to complete;
    __syncthreads();

    //shared memory load operation;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

/*
5.2.1.3 动态共享内存
可以动态声明共享内存，从而实现这些相同 kernel 函数。
可以在kernel函数外声明动态共享内存，将其作用域限制在该kernel之中。
动态共享内存必须被声明为一个未定大小的一维数组，因此基于二维线程索引来计算内存访问索引。
因为要再这个kernel函数中按行主序写入，按列主序读取，所以需要保留以下两个索引：
row_idx: 根据二维线程索引计算出的一维行主序内存偏移量
col_idx: 根据二维线程索引计算出的一维列主序内存偏移量
使用已经计算的 row_idx, 按行主序写入共享内存，如下所示：
tile[row_idx] = row_idx;

在共享内存块被填满之后，使用适当的同步，然后按列主序将其读出并分配到全局内存中，如下所示：
out[row_idx] = tile[col_idx];

因为 out 数组存储在全局内存中，并且线程按行主序被安排在一个线程块内，所以为了确保合并存储，
需要通过线程坐标按行主序对 out 数组写入。该 kernel 函数的代码如下：
*/

__global__ void setRowReadColDyn(int *out){
    // dynamic shared memory
    extern __shared__ int tile[];

    //mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    //shared memory store operation
    tile[row_idx] = row_idx;
    
    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}

/*
5.2.1.4 填充静态声明的共享内存
正如章节 5.1.3.4 所描述的，填充数组是避免存储体冲突的一种方法。填充静态声明的
共享内存很简单。只需简单地将一列添加到二维共享内存分配中，代码如下所示：
*/
__global__ void setRowReadColPad(int *out){
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX+IPAD];

    // mapping from thread index to global memory offset;
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    //wait for all threads to complete
    __syncthreads();

    //shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];

}

/*
5.2.1.5 填充动态声明的共享内存
填充动态声明的共享内存数组更加复杂。当执行从二维线程索引到一维内存索引的转换时，对于每一行必须跳过一个填充的内存空间，
代码如下：
unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

因为在以下的kernel 函数中用于存储数据的全局内存小于填充的共享内存，所以需要三个索引：
一个索引用于按照行主序写入共享内存，一个索引用于按照列主序读取共享内存，
一个索引用于未填充的全局内存的合并访问，代码如下所示：
*/
__global__ void setRowReadColDynPad(int *out){
    //dynamic shared memory
    extern __shared__ int tile[];

    //mapping from thread index to global memory index;
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
    // shared memory store operation
    tile[row_idx] = g_idx;
    
    //wait for all threads to complete
    __syncthreads();

    //shared memory load operation
    out[g_idx] = tile[col_idx];

}

void printData(char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}

int main(int argc,char **argv){
    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("%s at",argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    cudaDeviceGetSharedMemConfig(&pConfig);
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    //setup array with size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block(BDIMX,BDIMY);
    dim3 grid(1,1);

    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
           block.y);

    // allocate device memory
    //需要先准备全局内存
    int *d_C;
    CHECK(cudaMalloc((int **)&d_C,nBytes));
    int *gpuRef;
    gpuRef = (int *)malloc(nBytes);

    CHECK(cudaMemset(d_C,0,nBytes));
    setColReadCol<<<grid,block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set col read col   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read row   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    //在启动内核时，必须指定共享内存的大小：
    setRowReadColDyn<<<grid,block,BDIMX*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col dyn", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col pad", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    //在启动内核时，必须指定共享内存的大小：
    setRowReadColDynPad<<<grid,block,(BDIMX+IPAD)*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col DP ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(cudaFree(d_C));
    free(gpuRef);
 
    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/*
在 V100 GPU 上，具有 4 字节共享内存访问模式的结果如下，它们清楚地展示了按行访问共享内存以提升性能，
因为相邻线程引用相邻字：
 Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   30.24%  4.8000us         1  4.8000us  4.8000us  4.8000us  setColReadCol(int*)
                   25.60%  4.0640us         2  2.0320us  1.9520us  2.1120us  [CUDA memcpy DtoH]
                   23.39%  3.7120us         2  1.8560us  1.7920us  1.9200us  [CUDA memset]
                   20.77%  3.2960us         1  3.2960us  3.2960us  3.2960us  setRowReadRow(int*)

接下来，可以使用nvprof中的指标以检查存储体冲突：
shared_load_transactions_per_request
shared_store_transactions_per_request

nvprof 结果如下，这些结果表明，在setRowReadRow kernel 函数中，线程束的存储和加载请求由一个事务完成。
而相同的请求在 setColReadCol kernel 函数中由 32 个事务完成。这证明在 Kepler 设备上，当使用 4 字节共享内存存储体时，
kernel 函数会有 32 路存储体冲突：

==123042== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Kernel: setColReadCol(int*)
shared_load_transactions_per_request   Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   32.000000   32.000000   32.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

类似地，在使用按行主序写，按列主序读的机制后，指标如下：
 Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   25.52%  5.9200us         3  1.9730us  1.6640us  2.2720us  [CUDA memcpy DtoH]
                   23.59%  5.4720us         3  1.8240us  1.7920us  1.8560us  [CUDA memset]
                   20.69%  4.8000us         1  4.8000us  4.8000us  4.8000us  setColReadCol(int*)
                   16.14%  3.7440us         1  3.7440us  3.7440us  3.7440us  setRowReadCol(int*)
                   14.07%  3.2640us         1  3.2640us  3.2640us  3.2640us  setRowReadRow(int*)

==4337== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   32.000000   32.000000   32.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

Store 操作是无冲突的，但是 load 操作显示出 32 路的冲突。 

在使用动态一维的共享内存时：
GPU activities:   24.34%  7.4240us         4  1.8560us  1.6640us  2.1120us  [CUDA memcpy DtoH]
                   24.24%  7.3920us         4  1.8480us  1.7910us  1.9840us  [CUDA memset]
                   15.64%  4.7680us         1  4.7680us  4.7680us  4.7680us  setColReadCol(int*)
                   12.48%  3.8070us         1  3.8070us  3.8070us  3.8070us  setRowReadCol(int*)
                   12.28%  3.7440us         1  3.7440us  3.7440us  3.7440us  setRowReadColDyn(int*)
                   11.02%  3.3600us         1  3.3600us  3.3600us  3.3600us  setRowReadRow(int*)
==103845== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.562500   32.562500   32.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   32.000000   32.000000   32.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

性能与使用静态内存的setRowReadCol类似，但是使用了一维索引计算出的动态声明的共享内存。

添加静态填充声明后，性能明显提升：
==64705== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   24.98%  9.2800us         5  1.8560us  1.7920us  2.0160us  [CUDA memset]
                   24.46%  9.0880us         5  1.8170us  1.6320us  2.1120us  [CUDA memcpy DtoH]
                   12.92%  4.8000us         1  4.8000us  4.8000us  4.8000us  setColReadCol(int*)
                   10.25%  3.8080us         1  3.8080us  3.8080us  3.8080us  setRowReadColDyn(int*)
                   10.17%  3.7770us         1  3.7770us  3.7770us  3.7770us  setRowReadCol(int*)
                    9.22%  3.4240us         1  3.4240us  3.4240us  3.4240us  setRowReadRow(int*)
                    8.01%  2.9760us         1  2.9760us  2.9760us  2.9760us  setRowReadColPad(int*)

用 nvprof 检查这个 kernel 函数的内存事务，结果如下所示：
==64833== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.562500   32.562500   32.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   32.000000   32.000000   32.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
对于 4-字节 模式的共享内存存储体而言，需要增加一列来解决存储体冲突。
对于 8-字节 模式的共享内存存储体而言，并非总是如此。
对于 8-字节 模式，每行需要填充的数据元素数量取决于二维共享内存的大小。
因此，在 8-字节 模式中，需要进行更多的测试。以便为 64 位模式确定合适的填充数量元素。    

在使用动态内存填充的情况下，性能与静态填充的情况类似：
==118923== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   25.07%  10.912us         6  1.8180us  1.6320us  2.2080us  [CUDA memcpy DtoH]
                   25.07%  10.912us         6  1.8180us  1.7920us  1.9200us  [CUDA memset]
                   11.03%  4.8000us         1  4.8000us  4.8000us  4.8000us  setColReadCol(int*)
                    8.75%  3.8080us         1  3.8080us  3.8080us  3.8080us  setRowReadColDyn(int*)
                    8.68%  3.7760us         1  3.7760us  3.7760us  3.7760us  setRowReadCol(int*)
                    7.57%  3.2960us         1  3.2960us  3.2960us  3.2960us  setRowReadRow(int*)
                    6.99%  3.0400us         1  3.0400us  3.0400us  3.0400us  setRowReadColPad(int*)
                    6.84%  2.9760us         1  2.9760us  2.9760us  2.9760us  setRowReadColDynPad(int*)

用 nvprof 检查这个 kernel 函数的内存事务，结果如下所示：      
==122434== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadColDynPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.562500   32.562500   32.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   32.593750   32.593750   32.593750
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   32.000000   32.000000   32.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000    

请注意，这些结果和填充静态声明的共享内存是一致的，所以这两种类型的共享内存可以被有效地填充。

到目前为止，从所有执行过的内核运行时间可以看出：
1. 使用填充的内核可提高性能，因为它减少了存储体冲突；
2. 带有动态声明共享内存的内核增加了少量的消耗。

要想显式每个内核产生的二维矩阵的内容，首先要将共享内存块的维度减小到 4，使其可以更简单地可视化：
# define BDIMX 4
# define BDIMX 4
输出如下：

$./smemSquare 1
./smemSquare atdevice 0: Tesla V100-SXM2-16GB with Bank Mode:4-Byte <<< grid (1,1) block (4,4)>>>
set col read col   :     0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
set row read row   :     0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
set row read col   :     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
set row read col dyn:     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
set row read col pad:     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
set row read col DP :     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15

根据结果可以看出，如果读和写操作使用不同的顺序（例如，读操作使用行主序，而写操作使用列主序），
那么 kernel 函数会产生转置矩阵。这些简单的kernel函数为更复杂的转置算法奠定了基础。
*/



