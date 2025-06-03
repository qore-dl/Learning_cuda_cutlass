#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMX 16
/*
数组大小由下面的宏设置为4：
*/
#define SEGM  4

/*
5.6 线程束洗牌指令
在本章中，已经介绍了如何使用共享内存执行线程块中线程间的低延迟数据交换。
从用 Kepler 系列的 GPU （计算能力为 3.0 或更高）开始，洗牌指令（shuffle instruction）
作为一种机制被加入其中，只要两个线程在相同的线程束中，那么就允许这两个线程直接读取另一个线程的寄存器。

洗牌指令使得线程束中的线程彼此之间可以直接交换数据，而不是通过共享内存或全局内存来进行的。
洗牌指令比共享内存有更低的延迟，并且该指令在执行数据交换时不消耗额外的内存。
因此，洗牌指令为应用程序快速交换线程束中线程间的数据提供了一个有吸引力的方法。

因为洗牌指令在线程束中的线程之间被执行，所以首先介绍一下束内线程（lane）的概念。
简单来说，一个束内线程指的是线程束内的单一线程。线程束中的每个束内线程被[0,31]范围内
线程索引（lane index）唯一标识。线程束中的每个线程都有一个都有一个唯一的束内线程索引，
并且同一线程块中的多个线程可以有相同的束内线程索引（就像一个网格中的多个线程可以有相同的threadIdx.x值一样）。
然而，束内线程索引没有内置变量，因为线程索引有内置变量。在一维线程块中，对于一个给定线程的束内线程索引和
线程束索引可以按以下公式进行计算：
laneID = threadIdx.x % 32
warpID = threadIdx.x / 32

例如，线程块中的线程 1 和线程 33 都有束内线程 ID 1，但它们有不同的线程束 ID。
对于二维线程块，可以将二维线程坐标转换为一维线程索引，并应用前面的公式来确定
束内线程和线程束的索引。
*/

/*
5.6.1 线程束洗牌指令的不同形式
有两组洗牌指令：一组用于整型变量，另一组用于浮点型变量。每组有 4 种形式的洗牌指令。
在线程束内交换整型变量，其基本函数标记如下：
int __shfl(int var,int srcLane,int width=warpSize);
内部指令 __shfl 返回值是var，var 通过由 srcLane 确定的同一线程束中的线程传递给__shfl。
srcLane 的含义变化取决于宽度值。
这个函数能使线程束中的每个线程都可以直接从一个特定的线程中获取某个值。
线程束内所有活跃的线程都同时产生此操作，这将导致每个线程中有4字节数据的移动。

变量 width 可以被设置为 2~32 之间 2 任何的指数（包括 2 和 32），这是可以选择的。
当设置为默认的 warpSize (即 32) 时，洗牌指令跨整个线程束执行，并且srcLane 指定源线程的束内线程索引。
然而，设置 width 允许将线程束细分为段，使每段包含有 width 个线程，并且在每个段上执行独立的洗牌操作。
对于不是 32 的其他 width 值，线程的束内线程 ID 和其在洗牌操作中的 ID 不一定相同。
在这种情况下，一维线程块中的线程洗牌 ID 可以按以下公式进行计算：
shuffle_ID = threadIdx.x % width;
例如，如果 shfl 被线程束中的每个线程通过以下参数调用：
int y = shfl(x,3,16);
那么线程 0~15将从线程 3 接收 x 的值，线程 16~31 将从线程 19 接收 x 的值（在线程束的后16个线程中其偏移量为3）
为了简单起见，srcLane 将被称为在本节的其余部分提到过的束内线程索引。

当传递给 shfl 的束内线程索引在线程束中所有线程调用时相同，
指令从特定的束内线程到线程束中所有线程都执行线程束广播操作：
srcLane: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31

线程束索引: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31
_shfl(val,2,32): 一条从束内线程 2 到线程束中所有线程的广播

洗牌操作的另一种形式是从与调用线程相关的线程中复制数据：
int __shfl_up(int var,unsigned int delta,int width=wrapSize)
__shfl_up 通过减去调用的束内线程索引 delta 来计算源束内线程索引。
返回由源线程所持有的值。因此，这一指令通过束内线程delta 将var 右移到线程束其他线程中。
__shfl_up 周围没有线程束，所以线程束中最低的delta个线程将保持不变，如下图所示：
srcLane: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31

线程束索引: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31
__shfl_up(val,2,32): 将值转移给右边的两个束内线程中：0->2,1->3,2->4,3->5,...,28->30,29->31 (低两个线程(即0,1)中变量不变)
*/

/*
相反，洗牌指令的第三种形式是从相对于调用线程而言具有高索引的线程中复制：
int __shlf_down(int var,unsinged int delta,int width=warpSize)

__shlf_down 通过给调用的束内线程索引增加 delta来计算源束内线程索引。
返回由源线程持有的值。因此，该指令通过将束内线程的var值左移 delta 到线程束内相应的线程。
使用 __shfl_down 时周围没有线程束，所以线程束中最大的delta 个束内线程将保持不变，如下所示：
srcLane: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31

线程束索引: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31
__shfl_down(val,2,32): 将值转移给右边的两个束内线程中：31->29,30->28,29->27,28->26,...,3->1,2->0 (高两个线程(即31,30)中变量不变)
*/

/*
洗牌指令的最后一种形式是根据调用束内线程索引自身的按位异或来传输束内线程中的数据：
int __shfl_xor(int var,int laneMask,int width=warpSize)
通过使用laneMask 执行调用束内线程索引的按位异或，内部指令可以计算源束内线程索引。
返回由源线程持有的值。该指令适合于蝴蝶寻址模式（a butterfly addressing pattern）:
srcLane: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31

线程束索引: 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31
__shfl_xor(val,1):实现蝴蝶交换，0->1,1->0; 2->3;3->2;...,30->31,31->30;

在本节讨论的所有洗牌函数还支持单精度浮点值。浮点洗牌函数采用浮点型的var参数，并返回一个浮点数。
否则，浮点洗牌函数就与整型洗牌函数相同了。
*/

/*
5.6.2 线程束内的共享数据
在本节中，会介绍几个有关线程洗牌指令的例子，并说明线程束洗牌指令的有点。
洗牌指令将诶应用到以下 3 种整型变量类型中：
标量变量
数组
向量型变量

5.6.2.1 跨线程束值的广播
下面的内核实现了线程束级的广播操作。每个线程都有一个寄存器变量 value。
源束内线程由变量srcLane 指定，它等同于跨所有线程。
每个线程都直接从源线程复制数据：
*/
__global__ void test_shfl_broadcast(int *d_out,int *d_in,int const srcLane){
    int value = d_in[threadIdx.x];
    value = __shfl(value,srcLane,BDIMX); /*为了简单起见，使用有16个线程的一维线程块*/
    d_out[threadIdx.x] = value;
}

/*
5.6.2.2 线程束内上移
下面的内核实现了洗牌上移的操作。线程束中每个线程的源束内线程都是独一无二的。
并由它自身的线程索引减去delta 来确定：
*/
__global__ void test_shfl_up(int *d_out,int *d_in,int const delta){
    int value = d_in[threadIdx.x];
    value = __shfl_up(value,delta,BDIMX);
    d_out[threadIdx.x] = value;
}

/*
5.2.6.2.3 线程束下移
下面的内核实现了洗牌下移的操作。线程束中每个线程的源束内线程都是独一无二的。
并由它自身的线程索引加上delta来确定：
*/
__global__ void test_shfl_down(int *d_out,int *d_in, unsigned int const delta){
    int value = d_in[threadIdx.x];
    value = __shfl_down(value,delta,BDIMX);
    d_out[threadIdx.x] = value;
}

/*
5.6.2.4 线程束内环绕移动
下面的kernel函数实现了跨线程束的环绕移动操作。每个线程的源束内线程是不同的，并由它自身的束内线程索引加上偏移量来确定。
偏移量可以为正数也可以为负数：
*/
__global__ void test_shfl_wrap(int *d_out,int *d_in,int const offset){
    int value = d_in[threadIdx.x];
    value = __shfl(value,threadIdx.x + offset,BDIMX);
    d_out[threadIdx.x] = value;
}

/*
5.6.2.5 跨线程束的蝴蝶交换
下面的内核实现了两个线程之间的蝴蝶寻址模式，这是通过调用线程和线程掩码确定的。
*/
__global__ void test_shfl_xor(int *d_out,int *d_in,int const mask){
    int value = d_in[threadIdx.x];
    value = __shfl_xor(value,mask,BDIMX);
    d_out[threadIdx.x] = value;
    //调用掩码为 1 的内核将导致相邻线程交换它们的值
}

/*
5.6.2.6 跨线程束交换数组值
考虑内核中使用寄存器数组的情况，在这种情况下，我们若想要在线程束的线程间交换数据的某些部分，
则可以使用洗牌指令交换线程束中线程间的数组元素。
在下面的内核中，每个线程都有一个寄存器数组 value，其大小是 SEGM。
每个线程从全局内存d_in中读取数据块到 value 中，使用由掩码确定的相邻线程块交换该块。
然后将接收到的数据写回到全局内存数据 d_out 中。
*/
__global__ void test_shfl_xor_array(int *d_out,int *d_in,int const mask){
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for(int i=0;i<SEGM;i++){
        value[i] = d_in[idx+i];
    }

    value[0] = __shfl_xor(value[0],mask,BDIMX);
    value[1] = __shfl_xor(value[1],mask,BDIMX);
    value[2] = __shfl_xor(value[2],mask,BDIMX);
    value[3] = __shfl_xor(value[3],mask,BDIMX);

    for (int i=0;i<SEGM;i++){
        d_out[idx+i] = value[i];
    }

}

/*
5.6.2.7 跨线程束使用数组索引交换数值
在之前的内核中，通过洗牌操作交换的数组元素在每个线程的本地数组中有相同的偏移量。
如果想在两个线程各自的数组中以不同的偏移量交换它们之间的元素，需要有基于洗牌指令的交换函数。
下面的函数交换了两个线程之间的一对值。布尔变量 pred 被用于识别第一个调用的线程，
它是交换数据的一对线程。要交换的数据元素由第一个线程的firstIdx 和第二个线程的 secondIdx 偏移标识的。
第一个调用线程通过交换firstIdx 和 secondIdx 中的元素开始，但此操作仅限于本地数组。
然后在两线程间的secondIdx 位置执行蝴蝶交换。最后，第一个线程交换接收自secondIdx 返回到 firstIdx 的元素：
*/

__inline__ __device__ void swap(int *value,int laneIdx,int mask,int firstIdx,int secondIdx){
    bool pred = ((laneIdx/mask + 1)==1);
    if(pred){
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }

    value[secondIdx] = __shfl_xor(value[secondIdx],mask,BDIMX);

    if(pred){
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

/*
下面的内核基于上述的交换函数，交换两个线程间不同偏移的两个元素：
*/
__global__ void test_shfl_swap(int *d_out,int *d_in,int const mask,int firstIdx;int secondIdx){
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for(int i=0;i<SEGM;i++){
        value[i] = d_in[idx + i];
    }

    swap(value,threadIdx.x,mask,firstIdx,secondIdx);

    for(int i=0;i<SEGM;i++){
        d_out[idx+i] = value[i];
    }
}

int main(int argc, char **argv)
{
    int dev = 0;
    bool iPrintout = 1;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting.", argv[0]);
    printf("at Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = BDIMX;
    int h_inData[BDIMX], h_outData[BDIMX];

    for (int i = 0; i < nElem; i++) h_inData[i] = i;

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    size_t nBytes = nElem * sizeof(int);
    int *d_inData, *d_outData;
    CHECK(cudaMalloc((int**)&d_inData, nBytes));
    CHECK(cudaMalloc((int**)&d_outData, nBytes));

    CHECK(cudaMemcpy(d_inData, h_inData, nBytes, cudaMemcpyHostToDevice));

    int block = BDIMX;

    // shfl bcast
    /*
    调用广播内核的方法如下，通过第三个参数，test_shfl_broadcast 将源束内线程设置为每个线程束内的第三个线程。
    全局内存的两片被传递给内核：输入数据和输出数据。
    */
    test_shfl_broadcast<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl bcast\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, -2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl wrap right\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl up
    /*
    线程束内上移调用如下，指定delta 为 2，其结果是将每个线程的值向右移动两个束内线程，
    最左边的两个束内线程值保持不变。
    */
    test_shfl_up<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl up \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    /*
    这个内核实现了环绕式左移操作。
    不同于由test_shfl_down产生的结果，最右边的两个束内线程的值也变化了。
    */
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl wrap left\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl offset
    /*
    通过指定一个负偏移量来调用内核
    这个内核实现了环绕式右移操作。此测试类似于test_shfl_up函数，不同的是这里最左边的两个束内线程也发生了改变。
    */
    test_shfl_wrap<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl wrap 2\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl down
    /*
    通过指定delta=2 进行洗牌下移的核函数调用。
    每个线程的值向左移动两个束内线程，最右边的两个束内线程保持不变。
    */
    test_shfl_down<<<1, block>>>(d_outData, d_inData, 2);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl down \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor
    //调用掩码为 1 的内核将导致相邻线程交换它们的值
    test_shfl_xor<<<1, block>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl xor 1\t\t: ");
        printData(h_outData, nElem);
    }

    test_shfl_xor<<<1, block>>>(d_outData, d_inData, -8);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl xor -1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - int4
    test_shfl_xor_int4<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl int4 1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - register array
    //因为每个线程有4个元素，所以线程块被缩小到原来大小的1/4。调用 kernel 函数如下所示：
    test_shfl_xor_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    /*
    掩码被设置为1，所以相邻的线程交换其数组值
    */
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl array 1\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - test_shfl_xor_element
    test_shfl_xor_element<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl idx \t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    test_shfl_xor_array_swap_base<<<1, block / SEGM>>>(d_outData, d_inData, 1,
            0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl swap base\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    /*
    通过指定掩码为1、第一个索引为0、第二个索引为3调用内核
    */
    test_shfl_xor_array_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl swap 0 3\t\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - swap
    test_shfl_swap<<<1, block / SEGM>>>(d_outData, d_inData, 1, 0, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("shfl swap inline\t: ");
        printData(h_outData, nElem);
    }

    // shfl xor - register array
    test_shfl_array<<<1, block / SEGM>>>(d_outData, d_inData, 1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost));

    if(iPrintout)
    {
        printf("initialData\t\t: ");
        printData(h_inData, nElem);
    }

    if(iPrintout)
    {
        printf("shfl array \t\t: ");
        printData(h_outData, nElem);
    }

    // finishing
    CHECK(cudaFree(d_inData));
    CHECK(cudaFree(d_outData));
    CHECK(cudaDeviceReset();  );

    return EXIT_SUCCESS;
}


