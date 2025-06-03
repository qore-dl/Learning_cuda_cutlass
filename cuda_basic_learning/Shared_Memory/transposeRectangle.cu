#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>

// Some kernels assume square blocks
#define BDIMX 16
#define BDIMY 32

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

#define IPAD 2

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%3.0f ", in[i]);
    }

    printf("\n");
    return;
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int index = INDEX(i, j, cols);
            if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
                match = 0;
                printf("different on (%d, %d) (offset=%d) element in "
                        "transposed matrix: host %f gpu %f\n", i, j, index,
                        hostRef[index], gpuRef[index]);
                break;
            }
        }
        if (!match) break;
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nrows, const int ncols)
{
    for (int iy = 0; iy < nrows; ++iy)
    {
        for (int ix = 0; ix < ncols; ++ix)
        {
            out[INDEX(ix, iy, nrows)] = in[INDEX(iy, ix, ncols)];
        }
    }
}

/*
5.4 合并的全局内存访问
使用共享内存也能避免对未合并的全局内存的访问。
矩阵转置就是一个典型的例子：读操作被自然合并，但写操作是按照交叉访问的。
在第4章中已表明，交叉访问是全局内存中最糟糕的访问方式，因为它浪费总线带宽。
在共享内存的帮助下，可以先在共享内存中进行转置操作，然后再对全局内存进行合并写操作。

在本章前面的部分，测试了一个矩阵转置kernel函数，该函数使用简单线程块对共享内存中的矩阵行进行写入，
并读取共享内存中的矩阵列。在本节中，将扩展 kernel 函数，具体方法是使用多个线程块对基于交叉的全局内存访问重新排序到合并访问。

5.4.1 基准转置内核
作为基准，下面的kernel函数是一个仅使用全局内存的矩阵转置的朴素实现。
*/

__global__ void naiveGmem(float *out, float *in, const int nrows, const int ncols)
{
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
        out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
    }
}

/*
因为 ix 是这个 kernel 函数二维线程配置的最内层维度，全局内存读操作在线程束内是被合并的，
而全局内存写操作在相邻线程间是交叉访问的。naiveGmem kernel 函数的性能是一个下界，
本节中涵盖的逐步优化在此被测量。

以执行合并访问为目的的更改写操作会生成副本内核。因为读写操作将被合并，但仍然执行相同数量的 I/O，
所以 copyGmem 函数将成为一个性能近似的上界：
*/



__global__ void copyGmem(float *out, float *in, const int nrows, const int ncols)
{
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
		    // NOTE this is a transpose, not a copy
        out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
    }
}

/*
5.4.2 使用共享内存的矩阵转置
为了避免交叉全局内存访问，可以使用二维共享内存来缓存原始矩阵中的数据。
从二维共享内存中读取的一列可以被转移到转置矩阵行中，他被存储在全局内存中。
虽然不是实现将导致共享内存存储体的冲突，但这个结果将比非合并的全局内存访问好得多
下面的kernel函数实现了使用共享内存的矩阵转置。它可以被看作是前面的章节中所讨论的
setRowReadCol的扩展。这两个kernel函数之间的差别在于setRowReadCol使用一个
线程块处理输入矩阵的单块转置，而transposSmem 扩展了转置操作，使用了多个线程块和多个数据块。
*/

__global__ void transposeSmem(float *out,float *in,int nx,int ny){
    //static shared memory
    __shared__ static float tile[BDIMY][BDIMX];
    /*
    kerneltransposeSmem 函数可以被分解为以下几个步骤：
    */

    //coordinate in original matrix
    unsigned int ix,iy,ti,to;
    /*
    对于每一个线程，若要想从全局内存和共享内存中取得正确的数据，都必须计算多个索引。
    对于一个给定的线程，首先要基于其线程索引和块索引计算器原始矩阵坐标，如下所示：
    */
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;

    //linear global memory index for original matrix
    //然后可以计算出全局内存的索引：
    ti = iy*nx + ix;
    /*
    因为 ix 是沿着线程块的最内层维度，包含 32 个线程的线程束可以用ti对全局内存进行行合并读取。
    */

    //thread index in transposed block
    unsigned int bidx,irow,icol;
    bidx = threadIdx.y*blockDim.x + threadIdx.x;
    /*
    此外，两个新的变量icol和irow被引入以代替 threadIdx。
    这些变量是相应转置块的索引：
    */
    irow = bidx/blockDim.y;
    icol = bidx%blockDim.y;

    /*
    类似地，转置矩阵的坐标计算公式如下所示：
    */
    //coordinate in transposed matrix
    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;

    /*
    与原始矩阵中线程的坐标计算相比，它有两个主要的差异：
    首先，转置矩阵中块的偏移交换了blockDim 和 blockIdx 的使用：
    线程配置的 x 维度被用来计算转置矩阵的列坐标，
    y 维度被用来计算转置矩阵的行坐标

   
    */

    //linear global memory index for transposed matrix;
    /*
    然后，用于存储转置矩阵的全局内存索引可以根据下列公式进行计算：
    */
    to = iy*ny+ix;

    //transpose with boundary test
    if (ix < nx && iy < ny){
        // load data from global memory to shared memory
        /*
        1. 线程束执行合并读取一行，该行存储在全局内存中的原始矩阵块中
        2. 然后，该线程束按行主序将该数据写入共享内存中，因此，这个写操作没有存储体冲突：
        利用计算出的偏移量，线程中的线程束可以从全局内存中连续读取，并对二维共享内存数据的tile的行进行写入
        */
        tile[threadIdx.y][threadIdx.x] = in[ti];
        /*
        3.因为线程块的读/写操作是同步的，所以会有一个填满全局内存数据的二维共享内存数组
        */
        //thread synchronization
        __syncthreads();
        /*
        4. 该线程束从二维共享内存数据中读取一列。由于共享内存没有被填充，所以会发生存储体冲突。
        5. 然后该线程束执行数据的合并写入操作，将其写入到全局内存的转置矩阵中的某行。
        */
        //store data to global memory from shared memory
        out[to] = tile[icol][irow];
        /*
        全局内存的读操作是合并的，同时共享内存存储体中的写操作没有冲突，因为每个线程束沿着tile中的一列读取数据。
        在本节的稍后部分将使用共享内存填充来解决存储体冲突问题。
        */
    }

}

/*
5.4.3 使用填充共享内存的矩阵转置
通过给二维共享内存数组tile中的每一行添加列填充，可以将原矩阵相同列中的数据元素均匀地划分到共享内存存储体中。
需要填充的列数取决于设备的计算能力和线程块大小。
对于一个大小为 32x16 的线程块被测试内核来说，在4-字节宽度模式下，需要增加两列填充：
__shared__ float tile[BDIMY][BDIMX+2];

此外，对tile 的store 和加载必须被转化以对每行中的额外两列负责。填充列会提供额外的加速
*/
__global__ void transposeSmemPad(float *out,float *in,int nx,int ny){
    //static shared memory with padding
    __shared__ float tile[BDIMY][BDIMX+IPAD];

    //coordinate in original matrix
    unsigned int ix,iy,ti,to;

    //coordinate in original matrix:
    iy = blockDim.y * blockIdx.y + threadIdx.y;
    ix = blockDim.x * blockIdx.x + threadIdx.x;

    ti = iy*nx + ix;

    unsigned int bidx,irow,icol;
    bidx = threadIdx.y*blockDim.x + threadIdx.x;

    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;

    to = iy*ny+ix;

    //transpose with boundary test
    if (ix < nx && iy < ny){
        // load data from global memory to shared memory
        /*
        1. 线程束执行合并读取一行，该行存储在全局内存中的原始矩阵块中
        2. 然后，该线程束按行主序将该数据写入共享内存中，因此，这个写操作没有存储体冲突：
        利用计算出的偏移量，线程中的线程束可以从全局内存中连续读取，并对二维共享内存数据的tile的行进行写入
        */
        tile[threadIdx.y][threadIdx.x] = in[ti];
        /*
        3.因为线程块的读/写操作是同步的，所以会有一个填满全局内存数据的二维共享内存数组
        */
        //thread synchronization
        __syncthreads();
        /*
        4. 该线程束从二维共享内存数据中读取一列。由于共享内存没有被填充，所以会发生存储体冲突。
        5. 然后该线程束执行数据的合并写入操作，将其写入到全局内存的转置矩阵中的某行。
        */
        //store data to global memory from shared memory
        out[to] = tile[icol][irow];
        /*
        全局内存的读操作是合并的，同时共享内存存储体中的写操作没有冲突，因为每个线程束沿着tile中的一列读取数据。
        在本节的稍后部分将使用共享内存填充来解决存储体冲突问题。
        */
    }
}

/*
5.4.4 使用展开的矩阵转置
下面的kernel 函数展开了两个数据块的同时处理：在每个线程现在转置了一个被数据块跨越的两个数据元素。
这种转化的目标是通过创造更多的同时加载和存储以提升设备内存带宽利用率：
*/

__global__ void transposeSmemUnrollPad(float *out,float *in, const int nx, const int ny){
    //static 1D shared memory with padding
    __shared__ float tile[BDIMY*(BDIMX*2+IPAD)];

    //coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //linear global memory index for original matrix
    unsigned int ti = iy*nx + ix;

    //thread index in transposed block
    /*
    共享内存转置块中的新的线程索引计算如下：
    */
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    //coordinate in transposed matrix
    /*
    最后，转置矩阵中输出矩阵的坐标和被用来存储计算结果的全局内存中的相应索引计算如下：
    */
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = blockIdx.x * blockDim.x + irow;

    //linear global memory index for transposed matrix
    unsigned int to = iy2*ny + ix2;

    if(ix < nx && iy < ny){
        //load two rows from global memory to shared memory
        /*
        由于共享内存数据tile是一维的，所以必须将二维线程索引转换为一维共享内存索引，
        以访问填充的一维共享内存：
        */
        /*
        使用上面计算出的全局和共享内存的索引，每个线程读取全局内存一行中的两个数据元素
        */
        unsigned int row_idx = threadIdx.y * (blockDim.x*2+IPAD) + threadIdx.x;
        /*
        因为填充的内存不是用来存储数据的，所以计算索引时必须跳过填充列。
        */
        tile[row_idx] = in[ti];
        if((ix+blockDim.x)<nx && iy<ny){
            tile[row_idx+BDIMX] = in[ti+BDIMX];
        }
        //thread synchronization
        __syncthreads();

        //store two rows to global memory from two columns of shared memory
        /*
        并将它们写入到全局内存中的一行中。请注意，由于共享内存数组tile有添加填充，
        所以这些沿着同一列的共享内存请求不会导致存储体冲突：
        */
        unsigned int col_idx = icol * (blockDim.x*2+IPAD) + irow;
        out[to] = tile[col_idx];
        if((ix+blockDim.x)<nx && iy<ny){
            out[to+ny*BDIMX] = tile[col_idx+BDIMX];
        }
    }
    
}
//对这个kernel函数进行一个微小的扩展可以提供更多的灵活性，例如可以将共享内存数据tile的声明替换成动态声明模式
//以允许动态共享内存的分配。正如在以前的例子中观察到的，预计会有微小的性能下降：
__global__ void transposeSmemUnrollPadDyn(float *out,float *in, const int nx, const int ny){
    //static 1D shared memory with padding
    extern __shared__ float tile[];

    //coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //linear global memory index for original matrix
    unsigned int ti = iy*nx + ix;

    //thread index in transposed block
    /*
    共享内存转置块中的新的线程索引计算如下：
    */
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    //coordinate in transposed matrix
    /*
    最后，转置矩阵中输出矩阵的坐标和被用来存储计算结果的全局内存中的相应索引计算如下：
    */
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = blockIdx.x * blockDim.x + irow;

    //linear global memory index for transposed matrix
    unsigned int to = iy2*ny + ix2;

    if(ix < nx && iy < ny){
        //load two rows from global memory to shared memory
        /*
        由于共享内存数据tile是一维的，所以必须将二维线程索引转换为一维共享内存索引，
        以访问填充的一维共享内存：
        */
        /*
        使用上面计算出的全局和共享内存的索引，每个线程读取全局内存一行中的两个数据元素
        */
        unsigned int row_idx = threadIdx.y * (blockDim.x*2+IPAD) + threadIdx.x;
        /*
        因为填充的内存不是用来存储数据的，所以计算索引时必须跳过填充列。
        */
        tile[row_idx] = in[ti];
        if((ix+blockDim.x)<nx && iy<ny){
            tile[row_idx+BDIMX] = in[ti+BDIMX];
        }
        //thread synchronization
        __syncthreads();

        //store two rows to global memory from two columns of shared memory
        /*
        并将它们写入到全局内存中的一行中。请注意，由于共享内存数组tile有添加填充，
        所以这些沿着同一列的共享内存请求不会导致存储体冲突：
        */
        unsigned int col_idx = icol * (blockDim.x*2+IPAD) + irow;
        out[to] = tile[col_idx];
        if((ix+blockDim.x)<nx && iy<ny){
            out[to+ny*BDIMX] = tile[col_idx+BDIMX];
        }
    }
    
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool iprint = 0;

    // set up array size 4096
    int nrows = 1 << 12;
    int ncols = 1 << 12;

    if (argc > 1) iprint = atoi(argv[1]);

    if (argc > 2) nrows = atoi(argv[2]);

    if (argc > 3) ncols = atoi(argv[3]);

    printf(" with matrix nrows %d ncols %d\n", nrows, ncols);
    size_t ncells = nrows * ncols;
    size_t nBytes = ncells * sizeof(float);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    /*
     * Map CUDA blocks/threads to output space. Map rows in output to same
     * x-value in CUDA, columns to same y-value.
     */
    dim3 grid ((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
    dim3 grid2 ((grid.x + 2 - 1) / 2, grid.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nrows * ncols);

    //  transpose at host side
    transposeHost(hostRef, h_A, nrows, ncols);

    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    double iStart = seconds();
    copyGmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nrows * ncols);

    float ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) /
        iElaps;
    ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    printf("copyGmem elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    naiveGmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    printf("naiveGmem elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // // tranpose smem
    // CHECK(cudaMemset(d_C, 0, nBytes));
    // memset(gpuRef, 0, nBytes);

    // iStart = seconds();
    // naiveGmemUnroll<<<grid2, block>>>(d_C, d_A, nrows, ncols);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;

    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // if(iprint) printData(gpuRef, ncells);

    // checkResult(hostRef, gpuRef, ncols, nrows);
    // ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    // ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    // printf("naiveGmemUnroll elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
    //        "effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y, block.x,
    //        block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmem elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemPad<<<grid, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemPad elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // // tranpose smem pad
    // CHECK(cudaMemset(d_C, 0, nBytes));
    // memset(gpuRef, 0, nBytes);

    // iStart = seconds();
    // transposeSmemDyn<<<grid, block, BDIMX*BDIMY*sizeof(float)>>>(d_C, d_A, nrows,
    //         ncols);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;

    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // if(iprint) printData(gpuRef, ncells);

    // checkResult(hostRef, gpuRef, ncols, nrows);
    // ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    // ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    // printf("transposeSmemDyn elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
    //        "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
    //        block.y, ibnd);

    // // tranpose smem pad
    // CHECK(cudaMemset(d_C, 0, nBytes));
    // memset(gpuRef, 0, nBytes);

    // iStart = seconds();
    // transposeSmemPadDyn<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(float)>>>(
    //       d_C, d_A, nrows, ncols);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;

    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // if(iprint) printData(gpuRef, ncells);

    // checkResult(hostRef, gpuRef, ncols, nrows);
    // ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    // ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    // printf("transposeSmemPadDyn elapsed %f sec <<< grid (%d,%d) block "
    //        "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid.x, grid.y,
    //        block.x, block.y, ibnd);

    // // tranpose smem
    // CHECK(cudaMemset(d_C, 0, nBytes));
    // memset(gpuRef, 0, nBytes);

    // iStart = seconds();
    // transposeSmemUnroll<<<grid2, block>>>(d_C, d_A, nrows, ncols);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;

    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // if(iprint) printData(gpuRef, ncells);

    // checkResult(hostRef, gpuRef, ncols, nrows);
    // ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    // ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    // printf("transposeSmemUnroll elapsed %f sec <<< grid (%d,%d) block "
    //        "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y,
    //        block.x, block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemUnrollPad<<<grid2, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemUnrollPad elapsed %f sec <<< grid (%d,%d) block "
           "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y,
           block.x, block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemUnrollPadDyn<<<grid2, block, (BDIMX * 2 + IPAD) * BDIMY *
        sizeof(float)>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemUnrollPadDyn elapsed %f sec <<< grid (%d,%d) block "
           "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y,
           block.x, block.y, ibnd);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/*
在 V100 GOU 上，copyGmem 和 naiveGmem kernel 函数的结果总结如下：
$nvprof ./transposeRectangle
==45830== NVPROF is profiling process 45830, command: ./transposeRectangle
./transposeRectangle starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nrows 4096 ncols 4096
copyGmem elapsed 0.000276 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 486.139862 GB
naiveGmem elapsed 0.000455 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 294.892578 GB
==45830== Profiling application: ./transposeRectangle
==45830== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.04%  21.851ms         2  10.925ms  10.328ms  11.522ms  [CUDA memcpy DtoH]
                   34.82%  12.067ms         1  12.067ms  12.067ms  12.067ms  [CUDA memcpy HtoD]
                    1.16%  401.66us         1  401.66us  401.66us  401.66us  naiveGmem(float*, float*, int, int)
                    0.52%  180.48us         1  180.48us  180.48us  180.48us  copyGmem(float*, float*, int, int)
copyGmem 的速度是 naiveGmem的约2.25倍。由于朴素内核写入全局内存，使其带有了4096个元素的跨度，
所以单一线程束的存储操作是由 32 个全局内存事务完成的，可以使用nvprof 指标来确认这一点：
==74635== Profiling application: ./transposeRectangle
==74635== Profiling result:
==74635== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: copyGmem(float*, float*, int, int)
gld_transactions_per_request      Global Load Transactions Per Request    4.000000    4.000000    4.000000
gst_transactions_per_request     Global Store Transactions Per Request    4.000000    4.000000    4.000000
Kernel: naiveGmem(float*, float*, int, int)
gld_transactions_per_request      Global Load Transactions Per Request    4.000000    4.000000    4.000000
gst_transactions_per_request     Global Store Transactions Per Request   16.000000   16.000000   16.000000

2. 在使用共享内存暂存进行写合并的情况下，性能如下：
$nvprof ./transposeRectangle
==88925== NVPROF is profiling process 88925, command: ./transposeRectangle
./transposeRectangle starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nrows 4096 ncols 4096
copyGmem elapsed 0.000474 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 283.174011 GB
naiveGmem elapsed 0.000442 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 303.640747 GB
transposeSmem elapsed 0.000267 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 502.633881 GB
==88925== Profiling application: ./transposeRectangle
==88925== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   71.15%  33.005ms         3  11.002ms  10.223ms  11.947ms  [CUDA memcpy DtoH]
                   26.15%  12.128ms         1  12.128ms  12.128ms  12.128ms  [CUDA memcpy HtoD]
                    0.87%  402.24us         1  402.24us  402.24us  402.24us  naiveGmem(float*, float*, int, int)
                    0.86%  401.09us         1  401.09us  401.09us  401.09us  copyGmem(float*, float*, int, int)
                    0.51%  235.84us         3  78.613us  76.224us  83.264us  [CUDA memset]
                    0.46%  214.24us         1  214.24us  214.24us  214.24us  transposeSmem(float*, float*, int, int)
使用共享内存提高了转置内核的性能

使用nvprof 报告 transposeSmem 函数中每个请求执行全局内存事务数量：
==89309== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: copyGmem(float*, float*, int, int)
gld_transactions_per_request      Global Load Transactions Per Request    4.000000    4.000000    4.000000
gst_transactions_per_request     Global Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: naiveGmem(float*, float*, int, int)
gld_transactions_per_request      Global Load Transactions Per Request    4.000000    4.000000    4.000000
gst_transactions_per_request     Global Store Transactions Per Request   16.000000   16.000000   16.000000
Kernel: transposeSmem(float*, float*, int, int)
gld_transactions_per_request      Global Load Transactions Per Request    4.000000    4.000000    4.000000
gst_transactions_per_request     Global Store Transactions Per Request    4.000000    4.000000    4.000000

全局内存存储的重复数量从 16 减少到了4。
由于转置块张总的块宽为16，所以线程束前半部分的写操作和线程束前半部分的写操作和线程束后半部分的写操作
间隔了4080；因此，线程束的写入全局内存请求是由多个事务完成的。
将线程块大小更改到 32x32会把重复次数减少到 1。
然而，32x16的线程块配置比32x32的启动配置显示出了更多的并行性。
之后会调查哪个优化会更有利。
显然，读取二维共享内存数组中的一列会产生存储体冲突。在 4字节共享内存存储体模式下，会出现 16 个事务的重复，
在8字节宽度存储体模式下，会产生 8 路冲突：
==113589== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
Kernel: copyGmem(float*, float*, int, int)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
Kernel: naiveGmem(float*, float*, int, int)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
Kernel: transposeSmem(float*, float*, int, int)
shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.559710   16.559710   16.559710
shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.043707    1.043707    1.043707


在使用静态填充机制后，结果如下：
$nvprof ./transposeRectangle
==28798== NVPROF is profiling process 28798, command: ./transposeRectangle
./transposeRectangle starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nrows 4096 ncols 4096
copyGmem elapsed 0.000482 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 278.412445 GB
naiveGmem elapsed 0.000443 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 302.987061 GB
transposeSmem elapsed 0.000269 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 499.069122 GB
transposeSmemPad elapsed 0.000243 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 552.453369 GB
==28798== Profiling application: ./transposeRectangle
==28798== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   75.93%  42.493ms         4  10.623ms  10.130ms  11.587ms  [CUDA memcpy DtoH]
                   21.37%  11.959ms         1  11.959ms  11.959ms  11.959ms  [CUDA memcpy HtoD]
                    0.72%  402.11us         1  402.11us  402.11us  402.11us  copyGmem(float*, float*, int, int)
                    0.72%  400.61us         1  400.61us  400.61us  400.61us  naiveGmem(float*, float*, int, int)
                    0.56%  311.55us         4  77.888us  76.128us  82.784us  [CUDA memset]
                    0.38%  215.20us         1  215.20us  215.20us  215.20us  transposeSmem(float*, float*, int, int)
                    0.33%  183.90us         1  183.90us  183.90us  183.90us  transposeSmemPad(float*, float*, int, int)

显然使用填充机制有效提升了性能。
对于存储体冲突优化而言，结果如下：
==29435== Metric result:
Invocations                               Metric Name                             Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: copyGmem(float*, float*, int, int)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: naiveGmem(float*, float*, int, int)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    0.000000    0.000000    0.000000
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    0.000000    0.000000    0.000000
    Kernel: transposeSmemPad(float*, float*, int, int)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    2.289101    2.289101    2.289101
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    2.023270    2.023270    2.023270
    Kernel: transposeSmem(float*, float*, int, int)
          1      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request   16.560295   16.560295   16.560295
          1     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    1.044315    1.044315    1.044315

显然，有效缓解了在 shared memory load 时的存储体冲突现象。

在引入了展开技术后，可以观察到显著的性能改善：
./transposeRectangle starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nrows 4096 ncols 4096
copyGmem elapsed 0.000476 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 281.897827 GB
naiveGmem elapsed 0.000441 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 304.297272 GB
transposeSmem elapsed 0.000258 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 520.286438 GB
transposeSmemPad elapsed 0.000225 sec <<< grid (256,128) block (16,32)>>> effective bandwidth 596.345276 GB
different on (16, 64) (offset=65600) element in transposed matrix: host 18.700001 gpu 17.100000
Arrays do not match.

transposeSmemUnrollPad elapsed 0.000165 sec <<< grid (128,128) block (16,32)>>> effective bandwidth 813.511475 GB
different on (16, 32) (offset=65568) element in transposed matrix: host 2.800000 gpu 22.100000
Arrays do not match.

transposeSmemUnrollPadDyn elapsed 0.000161 sec <<< grid (128,128) block (16,32)>>> effective bandwidth 832.766174 GB
==59196== Profiling application: ./transposeRectangle
==59196== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   82.04%  63.596ms         6  10.599ms  10.030ms  12.340ms  [CUDA memcpy DtoH]
                   15.49%  12.007ms         1  12.007ms  12.007ms  12.007ms  [CUDA memcpy HtoD]
                    0.60%  463.71us         6  77.285us  76.063us  82.880us  [CUDA memset]
                    0.52%  401.98us         1  401.98us  401.98us  401.98us  copyGmem(float*, float*, int, int)
                    0.52%  401.89us         1  401.89us  401.89us  401.89us  naiveGmem(float*, float*, int, int)
                    0.28%  215.30us         1  215.30us  215.30us  215.30us  transposeSmem(float*, float*, int, int)
                    0.24%  184.03us         1  184.03us  184.03us  184.03us  transposeSmemPad(float*, float*, int, int)
                    0.16%  126.56us         1  126.56us  126.56us  126.56us  transposeSmemUnrollPadDyn(float*, float*, int, int)
                    0.16%  125.63us         1  125.63us  125.63us  125.63us  transposeSmemUnrollPad(float*, float*, int, int)

通过展开的两块，更多的内存请求将同时处于运行状态并且读/写吞吐量会提高，这可以通过nvprof来检查：
==59416== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: copyGmem(float*, float*, int, int)
          1                      dram_read_throughput             Device Memory Read Throughput  161.18GB/s  161.18GB/s  161.18GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  193.42GB/s  193.42GB/s  193.42GB/s
    Kernel: naiveGmem(float*, float*, int, int)
          1                      dram_read_throughput             Device Memory Read Throughput  161.50GB/s  161.50GB/s  161.50GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  193.91GB/s  193.91GB/s  193.91GB/s
    Kernel: transposeSmemPad(float*, float*, int, int)
          1                      dram_read_throughput             Device Memory Read Throughput  349.09GB/s  349.09GB/s  349.09GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  351.17GB/s  351.17GB/s  351.17GB/s
    Kernel: transposeSmemUnrollPadDyn(float*, float*, int, int)
          1                      dram_read_throughput             Device Memory Read Throughput  495.73GB/s  495.73GB/s  495.73GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  256.07GB/s  256.07GB/s  256.07GB/s
    Kernel: transposeSmem(float*, float*, int, int)
          1                      dram_read_throughput             Device Memory Read Throughput  293.23GB/s  293.23GB/s  293.23GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  294.28GB/s  294.28GB/s  294.28GB/s
    Kernel: transposeSmemUnrollPad(float*, float*, int, int)
          1                      dram_read_throughput             Device Memory Read Throughput  495.86GB/s  495.86GB/s  495.86GB/s
          1                     dram_write_throughput            Device Memory Write Throughput  256.13GB/s  256.13GB/s  256.13GB/s


*/

/*
5.4.5 增大并行性：
32 x 16 -> 16 x 16:

16 x 16:
transposeSmemUnrollPad elapsed 0.000167 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 803.066956 GB
different on (16, 16) (offset=65552) element in transposed matrix: host 5.500000 gpu 3.500000
Arrays do not match.

transposeSmemUnrollPadDyn elapsed 0.000158 sec <<< grid (128,256) block (16,16)>>> effective bandwidth 849.094971 GB

0.16%  125.41us         1  125.41us  125.41us  125.41us  transposeSmemUnrollPad(float*, float*, int, int)
0.16%  125.18us         1  125.18us  125.18us  125.18us  transposeSmemUnrollPadDyn(float*, float*, int, int)

全局内存与共享内存指标：
Kernel: transposeSmemUnrollPad(float*, float*, int, int)
                            gst_throughput                        Global Store Throughput  498.84GB/s  498.84GB/s  498.84GB/s
                            gld_throughput                         Global Load Throughput  498.84GB/s  498.84GB/s  498.84GB/s
      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.988403    1.988403    1.988403
     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    2.040102    2.040102    2.040102

Kernel: transposeSmemUnrollPadDyn(float*, float*, int, int)
                            gst_throughput                        Global Store Throughput  497.24GB/s  497.24GB/s  497.24GB/s
                            gld_throughput                         Global Load Throughput  497.23GB/s  497.23GB/s  497.23GB/s
      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    1.993752    1.993752    1.993752
     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    2.039818    2.039818    2.039818

32 x 16:
transposeSmemUnrollPad elapsed 0.000164 sec <<< grid (128,128) block (16,32)>>> effective bandwidth 818.241211 GB
different on (16, 32) (offset=65568) element in transposed matrix: host 2.800000 gpu 22.100000
Arrays do not match.

transposeSmemUnrollPadDyn elapsed 0.000158 sec <<< grid (128,128) block (16,32)>>> effective bandwidth 849.094971 GB

0.16%  126.21us         1  126.21us  126.21us  126.21us  transposeSmemUnrollPadDyn(float*, float*, int, int)
0.16%  126.14us         1  126.14us  126.14us  126.14us  transposeSmemUnrollPad(float*, float*, int, int)

16 x 16 性能略好，观察全局内存与共享内存指标：
Kernel: transposeSmemUnrollPadDyn(float*, float*, int, int)
                        gst_throughput                        Global Store Throughput  495.38GB/s  495.38GB/s  495.38GB/s
                        gld_throughput                         Global Load Throughput  495.38GB/s  495.38GB/s  495.38GB/s
    shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    2.785330    2.785330    2.785330
    shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    2.041515    2.041515    2.041515

Kernel: transposeSmemUnrollPad(float*, float*, int, int)
                            gst_throughput                        Global Store Throughput  497.28GB/s  497.28GB/s  497.28GB/s
                            gld_throughput                         Global Load Throughput  497.28GB/s  497.28GB/s  497.28GB/s
      shared_load_transactions_per_request    Shared Memory Load Transactions Per Request    2.780558    2.780558    2.780558
     shared_store_transactions_per_request   Shared Memory Store Transactions Per Request    2.042158    2.042158    2.042158

性能近似，但是因为16 x 16 增加了并行性，所以提升了全局内存吞吐量，且 16 x 16 取得了更优的存储体冲突减少。


*/

