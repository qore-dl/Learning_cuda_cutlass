/*
5.2.2 矩形共享内存

矩形共享内存是一个更普遍的二维共享内存，在矩阵共享内存中数组的行与列的数量不相等。

__shared__ int tile[Row][Col];

当执行一个转置操作时，不能像在方形共享内存中一样，只能通过简单地转换来引用矩形数组的线程坐标。
当使用矩形共享内存时，这样做会导致内存访问冲突。需要基于矩阵维度重新计算访问索引，
以重新实现之前描述的kernel函数。

一般情况下，需要测试一个矩阵共享内存数组，其每行有 32 个元素，每列有 16 个元素。
在下面的宏中定义了维度：
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMX 32
#define BDIMY 16
#define IPAD 2

void printData(char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%4d", in[i]);
        fflush(stdout);
    }

    printf("\n\n");
}

/*
矩形共享内存块被分配如下：
__shared__ int tile[BDIMY][BDIMX];

为了简单起见，内核将被启动为只有一个网格和一个二维线程块，该线程块的大小与矩阵共享内存数组相同，代码如下：
dim3 block (BDIMX,BDIMY);
dim3 grid (1,1);

5.2.2.1 行主序访问与列主序访问
将要测试的前两个 kernel 函数也在方形（共享内存）情况下使用：

__global__ void setRowReadRow(int *out);
__global__ void setColReadCol(int *out);

需要注意每个内核中矩形共享内存数组的声明。在 setRowReadRow kernel 函数中，
共享内存数据 tile 的最内层维度的长度被设置为同二维线程块最内层维度相同的长度：
__shared__ int tile[BDIMY][BDIMX];

在setColReadCol kernel 中，共享内存数组 tile 的最内层维度的长度被设置为同二维线程块的最外层维度相同的长度：
__shared__ int tile[BDIMX][BDIMY];

代码如下：
*/

__global__ void setRowReadRow(int *out){
    //static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    //wait for all threads complete
    __syncthreads();

    //shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out){
    //static shared memory
    __shared__ int tile[BDIMX][BDIMY];

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
5.2.2.2 行主序写操作和列主序读操作
在本节中，将实现一个kernel函数，该kernel函数使用一个矩形共享内存数组。
按行主序写入共享内存，并按列主序读取共享内存。
这个内核在现实的应用程序中是可用的。它使用共享内存执行矩阵转置，通过最大化低延迟的加载和存储来提高性能（shared memory）
并合并全局内存访问。
二维共享内存块被声明如下：
__shared__ int tile[BDIMY][BDIMX];

kernel 有 3 个内存操作：
1. 写入每个线程束的共享内存行，以避免存储体冲突；
2. 读取每个线程束中的共享内存列，以完成矩阵转置
3. 使用合并访问写入每个线程束的全局内存行
计算出正确的共享和全局内存访问的步骤如下所示。首先，将当前线程的二维线程索引转换为一维全局线程 ID：
unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

这个一维行主序的映射可以确保全局内存访问是合并的。因为输出的全局内存中的数据元素是转置过的，
所以需要计算转置矩阵中的新坐标，代码如下所示：
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;

通过将全局线程 ID 存储到二维共享内存块中来初始化共享内存块，如下：
tile[threaIdx.y][threadIdx.x] = idx;

此时，共享内存中的数据是从 0 到 BDIMX x BDIMY -1 线性存储的。
由于每个线程束对共享内存执行了行主序写入，因此在写操作期间没有存储体冲突。

现在，可以使用之前计算出的坐标访问转置过的共享内存数据。通过交换过的 irow 和 icol 访问共享内存，
可以用一维线程 ID 向全局内存写入转置数据。如下面的代码所示，线程束从共享内存的一列中读取数据元素，
并对全局内存执行合并写入操作：
out[idx] = tile[icol][irow];
完整的kernel 函数代码如下：
*/
__global__ void setRowReadCol(int *out){
    //static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    //mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //convert idx to transposed coordinate (row,col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    //shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    //wait for all threads to complete
    __syncthreads();

    //shared memory load operation
    out[idx] = tile[icol][irow];
}

/*
5.2.2.3 动态声明的共享内存
因为动态共享内存只能被声明为一维数组，当按照行写入和按照列读取时，将二维线程坐标转换为一维共享内存索引需要引入一个新的索引：
unsigned int col_idx = icol * threadIdx.x + irow;

因为icol 对应于线程块中最内层的维度，所以这种转换以列主序访问共享内存，这会导致存储体冲突。kernel 函数的代码如下：
*/
__global__ void setRowReadColDyn(int *out){
    //dynamic shared memory
    extern __shared__ int tile[];

    //mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //convert idx to transposed (row,col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    //convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * threadIdx.x + irow;

    //shared memory store operation
    tile[idx] = idx;
    
    //wait for all threads to complete
    out[idx] = tile[col_idx];

}

/*
5.2.2.4 填充静态声明的共享内存
对于矩阵共享内存，还可以使用共享内存填充来解决存储体冲突的问题。然而，对于 8字节宽度模式的存储体设置，
必须计算出需要多少填充元素。为了便于编程，使用宏来定义每一行添加的填充列的数量：
# define IPAD 2
填充的静态共享内存被声明如下：
__shared__ int tile[BDIMY][BDIMX+IPAD];
除添加了共享内存的填充以外,setRowReadColPad kernel 函数与 SetRowReadCol kernel 函数是相同的：
*/
__global__ void setRowReadColPad(int *out){
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX+IPAD];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    //convert idx to transposed (row,col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    //shared memory store operation;
    tile[threadIdx.y][threadIdx.x] = idx;

    //wait for all threads to complete
    __syncthreads();
    
    //shared memory load operation;
    out[idx] = tile[icol][irow];
}

/*
5.2.2.5
填充技术还可以用于动态共享内存的kernel中，该kernel使用矩形共享内存区域。
因为填充的共享内存和全局内存大小会有所不同，所以在内核中每个线程必须保留 3 个索引：
1. row_idx: 填充共享内存的行主序索引。使用该索引，线程束可以访问单一的矩阵行
2. col_idx: 填充共享内存的列主序索引。使用该索引，线程束可以访问单一的矩阵列
3. g_idx: 线性全局内存索引。使用该索引，线程束可以对全局内存进行合并访问。
这些索引是用以下代码计算出来的：
unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.y;
unsigned int irow = g_idx / blockDim.y;
unsigned int icol = g_idx % blockDim.y;
unsigned int row_idx = threadIdx.y *(blockDim.x+IPAD) + threadIdx.x;
unsigned int col_idx = icol*(blockDim.x+IPAD) + irow;

完整的 kernel 代码如下：
*/

__global__ void setRowReadColDynPad(int *out){
    //dynamic shared memory
    extern __shared__ int tile[];

    //mapping from thread index to global memory index
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    //convert idx to transposed (row,col)
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x+IPAD) + threadIdx.x;
    unsigned int col_idx = icol*(blockDim.x+IPAD) + irow;

    //shared memory store operation
    tile[row_idx] = g_idx;

    //wait for all threads complete
    out[g_idx] = tile[col_idx];
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
./smemRectangle atdevice 0: Tesla V100-SXM2-16GB with Bank Mode:4-Byte <<< grid (1,1) block (32,16)>>>
==68313== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   28.25%  3.9680us         2  1.9840us  1.9200us  2.0480us  [CUDA memcpy DtoH]
                   25.06%  3.5200us         1  3.5200us  3.5200us  3.5200us  setColReadCol(int*)
                   24.15%  3.3920us         2  1.6960us  1.5680us  1.8240us  [CUDA memset]
                   22.55%  3.1680us         1  3.1680us  3.1680us  3.1680us  setRowReadRow(int*)
使用 nvprof 指标检查存储体冲突的结果：
在 V100 GPU 上的结果显示如下：
==71532== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.625000   16.625000   16.625000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

共享内存的存储和加载请求，由 setRowReadRow kernel 函数中的 1 个事务完成。
同样的请求在 setColReadCol kernel 函数中由 16 个事务完成。
V100 存储体的宽度是4个字，一列 16 个 4 字节的数据元素被安排到 16 个存储体中。
因此，该操作有一个16路冲突。

考虑按行写按列读的例子后，使用nvprof 查看性能与内存事务：
==36613== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.91%  5.5360us         3  1.8450us  1.5680us  2.0160us  [CUDA memcpy DtoH]
                   24.26%  4.9920us         3  1.6640us  1.5680us  1.8560us  [CUDA memset]
                   17.11%  3.5200us         1  3.5200us  3.5200us  3.5200us  setColReadCol(int*)
                   16.33%  3.3600us         1  3.3600us  3.3600us  3.3600us  setRowReadRow(int*)
                   15.40%  3.1680us         1  3.1680us  3.1680us  3.1680us  setRowReadCol(int*)

==39178== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.562500   16.562500   16.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.625000   16.625000   16.625000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

store 操作是没有冲突的，但是load 操作会报告 16 路的冲突。
查看元素，发现均被转置了：
$./smemRectangle 1
./smemRectangle atdevice 0: Tesla V100-SXM2-16GB with Bank Mode:4-Byte <<< grid (1,1) block (4,2)>>>
set col read col   :    0   1   2   3   4   5   6   7

set row read row   :    0   1   2   3   4   5   6   7

set row read col   :    0   4   1   5   2   6   3   7

在使用了动态声明的共享内存设置时：
==92431== Profiling application: ./smemRectangle
==92431== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.37%  7.0720us         4  1.7680us  1.5360us  2.0480us  [CUDA memcpy DtoH]
                   25.18%  6.7520us         4  1.6880us  1.5680us  1.8880us  [CUDA memset]
                   13.37%  3.5840us         1  3.5840us  3.5840us  3.5840us  setColReadCol(int*)
                   11.93%  3.2000us         1  3.2000us  3.2000us  3.2000us  setRowReadCol(int*)
                   11.93%  3.2000us         1  3.2000us  3.2000us  3.2000us  setRowReadRow(int*)
                   11.22%  3.0080us         1  3.0080us  3.0080us  3.0080us  setRowReadColDyn(int*)
      API calls:   88.16%  339.43ms         1  339.43ms  339.43ms  339.43ms  cudaDeviceGetSharedMemConfig

性能与原来使用静态共享内存时基本一致。
使用nvprof 检查共享内存的事务时，报告如下：
==92706== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.562500   16.562500   16.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    4.125000    4.125000    4.125000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.625000   16.625000   16.625000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

在使用了静态填充的共享内存设置时：
==51379== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.70%  7.9360us         5  1.5870us  1.4080us  1.8880us  [CUDA memcpy DtoH]
                   25.40%  7.5520us         5  1.5100us  1.4400us  1.6640us  [CUDA memset]
                   10.66%  3.1680us         1  3.1680us  3.1680us  3.1680us  setColReadCol(int*)
                    9.69%  2.8800us         1  2.8800us  2.8800us  2.8800us  setRowReadCol(int*)
                    9.69%  2.8800us         1  2.8800us  2.8800us  2.8800us  setRowReadRow(int*)
                    8.93%  2.6560us         1  2.6560us  2.6560us  2.6560us  setRowReadColDyn(int*)
                    8.93%  2.6560us         1  2.6560us  2.6560us  2.6560us  setRowReadColPad(int*)

使用了静态的填充后，性能获得了明显的提升

用 nvprof 检查内存事务，结果报告如下：
==51631== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.562500   16.562500   16.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    4.125000    4.125000    4.125000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.625000   16.625000   16.625000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

实现了内存填充后，可以发现内存冲突明显减少了。
若使用 IPAD=1 的情况：
GPU activities:   26.37%  8.6410us         5  1.7280us  1.5360us  2.0810us  [CUDA memcpy DtoH]
                   25.09%  8.2230us         5  1.6440us  1.5670us  1.9520us  [CUDA memset]
                   10.84%  3.5520us         1  3.5520us  3.5520us  3.5520us  setColReadCol(int*)
                    9.77%  3.2000us         1  3.2000us  3.2000us  3.2000us  setRowReadCol(int*)
                    9.67%  3.1680us         1  3.1680us  3.1680us  3.1680us  setRowReadRow(int*)
                    9.18%  3.0080us         1  3.0080us  3.0080us  3.0080us  setRowReadColDyn(int*)
                    9.08%  2.9760us         1  2.9760us  2.9760us  2.9760us  setRowReadColPad(int*)
性能出现了下降。
用 nvprof 检查内存事务，结果报告如下：
==71657== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.562500   16.562500   16.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    4.125000    4.125000    4.125000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.625000   16.625000   16.625000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    2.000000    2.000000    2.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000

会发现 load 操作存在双向存储体冲突。

引入动态的内存填充机制后的性能：
==130078== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.73%  10.528us         6  1.7540us  1.5680us  2.0480us  [CUDA memcpy DtoH]
                   25.18%  9.9200us         6  1.6530us  1.5680us  1.8240us  [CUDA memset]
                    8.94%  3.5200us         1  3.5200us  3.5200us  3.5200us  setColReadCol(int*)
                    8.20%  3.2320us         1  3.2320us  3.2320us  3.2320us  setRowReadCol(int*)
                    8.04%  3.1680us         1  3.1680us  3.1680us  3.1680us  setRowReadRow(int*)
                    7.72%  3.0400us         1  3.0400us  3.0400us  3.0400us  setRowReadColDyn(int*)
                    7.72%  3.0400us         1  3.0400us  3.0400us  3.0400us  setRowReadColPad(int*)
                    7.47%  2.9440us         1  2.9440us  2.9440us  2.9440us  setRowReadColDynPad(int*)

用 nvprof 检查内存事务，结果报告如下：
==130181== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: setRowReadColDynPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.562500   16.562500   16.562500
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColDyn(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    4.125000    4.125000    4.125000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setColReadCol(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.625000   16.625000   16.625000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: setRowReadRow(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000
Kernel: setRowReadColPad(int*)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.000000    1.000000    1.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.000000    1.000000    1.000000  

性能与静态填充条件下类似。

在一般情况下，kernel函数使用共享内存填充消除存储体冲突以提高性能。
使用动态共享内存的kernel函数会显示处少量的消耗。
显示内容如下（BDIMX=8,BDIMX=2，小型矩阵方便展示）：
$./smemRectangle 1
./smemRectangle atdevice 0: Tesla V100-SXM2-16GB with Bank Mode:4-Byte <<< grid (1,1) block (8,2)>>>
set col read col   :    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15

set row read row   :    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15

set row read col   :    0   8   1   9   2  10   3  11   4  12   5  13   6  14   7  15

set row read col dyn:    0   1   1   4   2   7   3  10   4   5   5   8   6  11   7  14

set row read col pad:    0   8   1   9   2  10   3  11   4  12   5  13   6  14   7  15

set row read col DP :    0   8   1   9   2  10   3  11   4  12   5  13   6  14   7  15
有效实现了转置
*/
