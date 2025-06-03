#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIdx(void){
    printf("threadIdx: (%d,%d,%d) blockIdx: (%d,%d,%d) blockDim: (%d,%d,%d) gridDim:(%d,%d,%d)\n",
    threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
    gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc,char **argv){
    //define total data element
    int nElem = 6;

    //define grid and block structure
    dim3 block (3);
    dim3 grid((nElem+block.x-1)/block.x);
    /*
    给定数据大小，确定grid和block的尺寸的一般步骤为：
    1. 确定block的尺寸
    2. 在已知数据大小和块大小的基础上计算grid维度
    要确定block尺寸，主要考虑：
    1. kernel的性能特性
    2. GPU资源的限制
    
    */

    //check grid and block dimension from hot side

    printf("grid.x %d grid.y %d grod.z %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);

    /*
    区分host端和device端的grid和block变量的访问对于cuda代码非常重要，
    声明一个host端的block变量，可以按照如下定义其坐标并对其进行访问：
    block.x, block.y, block.z

    在device端，初始化kernel时预定义了内置block变量大小：
    blockDim.x,blockDim.y,blockDim.z

    总而言之，在启动kenrel之前就已经定义了host端的grid和block变量，并从host端，通过由x/y/z
    三个字段决定的矢量结构来访问它们。当kernel启动时，可以使用kernel中预初始化的内置变量来进行访问。
    */

    //check grid and block dimension from device side
    checkIdx<<<grid,block>>>();

    //reset device before you leave;
    cudaDeviceReset();

    return 0;
}