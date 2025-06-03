#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

/*
kernel函数是在device端执行的代码。
在kernel 函数中，需要为一个线程规定要进行计算以及要进行的数据访问
当kernel函数被调用时，许多不同的CUDA线程并行执行同一个计算任务

*/

__global__ void checkIdx(void){
    printf("threadIdx: (%d,%d,%d) blockIdx: (%d,%d,%d) blockDim: (%d,%d,%d) gridDim:(%d,%d,%d)\n",
    threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
    gridDim.x,gridDim.y,gridDim.z);
}

/*
kernel函数必须有一个void返回类型
CUDA C程序中的函数类型限定符：
函数类型限定符指定一个函数在主机上执行还是在设备上执行，以及可被host调用还是被device调用。

限定符          执行              调用                                                          备注
__global__    在device端执行    可以从host端调用也可以从计算能力>=3.0的device中调用         必须有一个void返回类型
__device__    在device端执行    仅能从device端调用
__host__      在host端执行      仅能从host端调用                                          限定符可以省略


__device__和__host__限定符可以一齐使用，这样函数可以同时在host和device端进行编译
*/

/*
CUDA kernel 函数（__global__）的限制（适用于所有kernel函数）
1. 只能访问device内存
2. 必须具有void返回类型
3. 不支持可变数量的参数
4. 不支持静态变量
5. 显示异步行为
 */





int main(int argc,char **argv){
    // define total data elements
    int nElem = 1024;

    //define grid and block structure
    dim3 block (1024);
    dim3 grid ((nElem+block.x-1)/block.x);
    printf("grid.x %d block.x %d \n",grid.x,block.x);
    checkIdx<<<grid,block>>>();
    CHECK(cudaDeviceSynchronize());

    //reset block
    block.x = 512;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x %d block.x %d \n",grid.x,block.x);
    checkIdx<<<grid,block>>>();
    CHECK(cudaDeviceSynchronize());

    //reset block
    block.x = 256;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x %d block.x %d \n",grid.x,block.x);
    checkIdx<<<grid,block>>>();
    CHECK(cudaDeviceSynchronize());

    //reset block
    block.x = 256;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x %d block.x %d \n",grid.x,block.x);
    //cuda的kernel调用是对C语言函数调用语句的延伸，<<<>>>运算符内是kernel函数的执行配置：
    // c_function_name (argument list) -> kernel_name <<<grid,block>>> (argument list);
    //CUDA编程模型揭示了线程层次结构，利用执行配置可以指定线程在GPU上调度运行的方式
    //执行配置的第一个值是grid维度，即启动的block的数量，第二个值是block的维度，即每个block中thread的数量
    //通过指定grid和block的维度，可以进行以下配置：
    //1. kernel中线程的数量
    //2. kernel中使用的线程布局。
    /*
    同一个block内部的thread之间可以相互协作，不同block之间的thread不能协作，
    对于一个给定的问题，
    可以使用不同的grid和block布局来组织线程的布局。
    例如，32个数据元素用于计算，8个元素一个block，启动4个block：
    一个block内线性布局了8个线程，不同block不同的组织

    由于数据在全局内存中是线性存储的，因此可以用变量blockIdx.x和threadIdx.x进行以下操作：
    1. 在grid中标识一个唯一的线程
    2. 建立线程和数据元素之间的映射关系

    若将32个元素都放到一个block里面，则只有一个block

    */
    checkIdx<<<grid,block>>>();
    CHECK(cudaDeviceSynchronize());
    /*
    核函数的的调用与主机线程是异步的。核函数调用结束后，控制权立刻返回给host端。
    可以调用以下函数来强制host端程序等待所有的核函数执行结束：
    cudaError_t cudaDeviceSynchronize(void)

    一些CUDA runtime API在host和device之间是隐式同步的。
    当使用cudaMemcpy函数在host和device之间拷贝数据时，host端隐式同步，
    即host端程序必须等待数据拷贝完成后才能继续执行程序。
    cudaError_t cudaMemcpy(void* dst,const void* src,size_t count,cudaMemcpykind kind);
    之前所有的kernel函数调用完成后开始拷贝数据，拷贝数据完成后，控制权立刻返回给host端
    */

    /*
    异步行为
    不同于C语言的函数调用，所有的CUDA kernel函数的启动都是异步的。CUDA内核调用完成后，控制权立刻返回给CPU。
    */

    //reset device before you leave;
    cudaDeviceReset();
    return 0;

    /*
    线程层次结构

    CUDA的特点之一就是通过编程模型揭示了一个两层的线程层次结构。
    由于一个kernel启动的grid和block的维数会影响性能，这一结构为程序员优化代码提供了一个额外的途径。

    grid和block的维度存在几个限制因素，对于block大小的一个主要限制因素就是可利用的计算资源，如寄存器，shared memory容量等
    某些限制可以通过查询GPU设备撤回

    grid和block从逻辑上代表了一个kernel函数的线程层次结构。这种线程组织方式可以使开发者在不同的设备上
    有效地执行相同的程序代码，而且每一个线程组织具有不同数量的计算和内存资源（如不同硬件配置下）

    */
}