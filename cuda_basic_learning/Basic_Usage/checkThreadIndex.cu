/*
组织并行线程

通过一个矩阵加法的例子来进一步说明线程组织的重要性。
对于矩阵运算而言，传统的方法是在 kernel中使用一个包含二维grid和二维block的布局来组织线程

但是，这种传统的方法无法获得最优性能，在矩阵加法中使用以下布局将有助于了解更多关于grid和block的启发性的用法：

由二维block构成的二维grid
由一维block构成的一维grid
由一维block构成的二维grid

使用block和thread建立矩阵索引

通常情况下，一个矩阵用行优先的方法在全局内存中进行线性存储（例如8x6,8个元素连续存储构成一行，共6行，每行在上一行之后开辟存储空间）
在一个矩阵加法kernel函数中，一个thread通常被分配一个数据元素处理。首先要完成的任务是使用block和thread从全局内存中访问指定的数据

通常情况下，对一个二维实例来说，需要管理3种索引：

1. thread和block索引
2. 矩阵中给定点的坐标
3.全局线性内存中的偏移量

对于一个给定的线程，首先可以通过把thread和block索引映射到矩阵坐标上，来获取block和thread索引的全局内存偏移量。
然后将这些矩阵坐标映射到全局内存的存储单元中。

第一步，可以用以下公式把thread和block索引映射到矩阵坐标上：
ix = threadIdx.x + blockIdx.x * blockDim.x;
iy = threadIdx.y + blockIdx.y * blockDim.y;

第二步，可以用以下公式把矩阵坐标映射到全局内存中的索引/存储单元上：
idx = iy*nx + ix;
 */

 /*
 printThreadInfo 函数被用于输出每个线程的以下信息：
 1. thread索引
 2. block索引
 3. 矩阵坐标
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 #include "common.h"

 void initialInt(int *ip,int size){
    for(int i=0;i<size;i++){
        ip[i] = i;
    }
 }

 void printMatrix(int *C,const int nx,const int ny){
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n",nx,ny);
    for (int iy=0;iy<ny;iy++){
        for (int ix=0;ix<nx;ix++){
            printf("%3d",ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
 }

 __global__ void printThreadIndex(int *A,const int nx, const int ny){

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;

    printf(
        "thread_id (%d,%d) block_id (%d,%d),coordinate(%d,%d) global index %2d ival %2d\n",
        threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
    
 }

 int main(int argc,char **argv){
    printf("%s Starting...\n",argv[0]);

    //get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //set matrix dimension;
    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);

    //malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);

    //initialize host matrix with integer
    initialInt(h_A,nxy);
    printMatrix(h_A,nx,ny);

    //malloc device memory
    int *d_MatA;
    cudaMalloc((void **)&d_MatA,nBytes);

    //transfer data from host to device
    cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);

    //set up execution configuration
    dim3 block(4,2);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    //invoke the kernel
    printThreadIndex<<<grid,block>>>(d_MatA,nx,ny);
    cudaDeviceSynchronize();

    //free host and device memory
    cudaFree(d_MatA);
    free(h_A);

    //reset device
    cudaDeviceReset();

    return 0;
 }