#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define RADIUS 4
#define BDIM 32

// constant memory
/*
在常量内存中声明coef数组，代码如下所示：
*/
__constant__ float coef[RADIUS+1];

// FD coeffecient
#define a0     0.00000f
#define a1     0.80000f
#define a2    -0.20000f
#define a3     0.03809f
#define a4    -0.00357f

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void printData(float *in,  const int size)
{
    for (int i = RADIUS; i < size; i++)
    {
        printf("%f ", in[i]);
    }

    printf("\n");
}



void cpu_stencil_1d (float *in, float *out, int isize)
{
    for (int i = RADIUS; i <= isize; i++)
    {
        float tmp = a1 * (in[i + 1] - in[i - 1])
                    + a2 * (in[i + 2] - in[i - 2])
                    + a3 * (in[i + 3] - in[i - 3])
                    + a4 * (in[i + 4] - in[i - 4]);
        out[i] = tmp;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int size)
{
    double epsilon = 1.0E-6;
    bool match = 1;

    for (int i = RADIUS; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                   gpuRef[i]);
            break;
        }
    }

    if (!match) printf("Arrays do not match.\n\n");
}

/*
5.5 常量内存
常量内存是一种专用的内存，它用于只读数据和统一访问线程束中线程的数据。
常量内存对内核代码而言是只读的，但它对于主机而言既是可读的也是可写的。
常量内存位于设备的DRAM上（和全局内存一样），并且有一个专用的片上缓存。
和L1 Cache 和共享内存一样，从每个 SM 的常量缓存中读取的延迟，比直接从常量内存中读取的低得多。
每个 SM 常量内存缓存大小的限制为 64 KB。
到目前为止，相较于在本书中学习的任何其他类型的内存而言，常量内存有一个不同的最优访问模式。
在常量内存中，如果线程束中的所有线程都访问相同的位置，那么这个访问模式就是最优的。
如果线程束中的线程访问不同的地址，则访问需要串行。
因此，一个常量内存读取的成本与线程束中线程读取唯一地址的数量呈线性关系。
在全局作用域中，必须用以下修饰符声明常量变量：
__constant__
常量内存变量的生存期与应用程序的生存期相同，其对网格内的所有线程都是可以访问的，
并且通过runtime 函数对主机可访问。
当使用 CUDA 独立编译能力时，常量内存变量跨多个源文件是可见的。
因为设备只能读取常量内存，所以在常量内存中的值必须使用以下运行时函数进行初始化：

cudaError_t cudaMemcpyToSymbol(const void *symbol,const void *src,size_t count,size_t offset,cudaMemcpyKind kind)

cudaMemcpyToSymbol 函数将src指向的数据复制到设备上由 symbol 指定的常量内存中。
枚举变量 kind 指定了传输方向，默认情况下，kind 是 cudaMemcpyHostToDevice。
*/

/*
5.5.1 使用常量内存实现一维模版
在数值分析中，模版计算在几何点集合上应用函数，并用输出更新单一点的值。
模版是求解许多偏微分方程算法的基础。在一维中，位置 x 周围的九点模版会给这些位置上的值应用一些函数：
{x-4h,x-3h,x-2h,x-h,x+h,x+2h,x+3h,x+4h}

图 5-18 展示了一个九点模版：
d_in[]: 0 |1 |2 |3 |4 |5 |6 |7 |8 |9 |10|11|12|13|14|15|
d_out:  0 |1 |2 |3 |4 |5 |6 |7 |8 |9 |10|11|12|13|14|15|

{4,5,6,7,8,9,10,11,12} -> {8}

一个九点模版的例子是实变量函数 f 在点 x 上一阶导数的第八阶中心差分公式。
理解这个公式的应用并不重要，只要简单的了解到它会将上述九点作为输入产生单一输出。
在本节中该公式将被作为一个示例模版。
f'(x) = c0(f(x+4h)-f(x-4h))+c1(f(x+3h)-f(x-3h))-c2(f(x+2h)-f(x-2h))+c3(f(x+h)-f(x-h))
在一维数据中对该公式的应用是对一个数据进行并行操作，该操作能很好地映射到 CUDA。
它可以为每个线程分配位置 x，并计算出f'(x)
*/

/*
现在，在模版计算中哪里可以应用常量内存？在上述模版公式的例子下，系数 c0、c1、c2 和 c3 在所有线程中都是相同的并且不会被修改。
这使得它们成为常量内存最优的候选，因为它们是只读的，并将呈现一个广播式的访问模式：线程束中的每个线程同时引用相同的常量内存地址。

下面的内核实现了基于上述公式的一维模版计算，由于每个线程需要9个点来计算1个点，所以要使用共享内存来缓存数据，从而减少对全局内存的冗余访问：
__shared__ float smem[BDIM + 2 * RADIUS];

RADIUS 定义了点 x 两侧点的数量，这些点被用于计算 x 点的值。在这个例子中，为了形成一个九点模版，RADIUS 被定义为4：
x 两侧各有 4 个点加上位置 x 的值。如图 5-19 所示，在每个块的左、右边界上各需要一个 RADIUS 个元素的光环：
        '        ' '           '
0|1|2|3|'4|5|6|7|'8'|9|10|11|12'|13|14|15|16|
        '        ' '           '
访问全局内存的索引可以使用以下语句来进行计算：
int idx = blockIdx * blockDim.x + threadIdx.x;
访问共享内存的每个线程的索引可使用以下语句来进行计算：
int sidx = threadIdx.x+RADIUS;
从全局内存中读取数据到共享内存中时，前4个线程负责从左侧和右侧的光环中读取数据到共享内存中，如下所示：
if(threadIdx.x < RADIUS){
    smem[sidx - RADIUS] = in[idx - RADIUS];
    smeme[sidx + BDIM] = in[idx+BDIM];
}
该模版计算是直接的。注意coef数组是存储上述系统的常量内存数组。
此外，#pragmaunroll 的作用是提示 CUDA 编译器，表明这个循环将被自动展开。
#pragma unroll
for(int i=1;i<=RADIUS;i++){
    tmp += coef[i] * (smem[sidx+i]-smem[sidx-i])
}

因为有限差分系数被存储在常量内存中，并且这是由主机线程准备的，所以在kernel函数中访问它们就访问数组一样简单。完整的kernel函数如下：
*/

__global__ void stencil_1d(float *in,float *out){
    //shared memory
    __shared__ float smem[BDIM+2*RADIUS];
    
    //index to global memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //index to shared memory for stencil calculation
    int sidx = threadIdx.x + RADIUS;

    //Read data from global memory into shared memory
    smem[sidx] = in[idx];

    //read halo part to shared memory
    if(threadIdx.x < RADIUS){
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    //synchronize (ensure al the data is available)
    __syncthreads();

    //Apply the stencil
    float tmp = 0.0f;
    #pragma unroll
    for(int i=1;i<=RADIUS;i++){
        tmp+=coef[i]*(smem[sidx+i] - smem[sidx-i]);
    }

    //store the result
    out[idx] = tmp;
}

/*
然后，使用cudaMemcpyToSymbol 的 CUDA API 调用从主机端初始化的常量内存
*/

void setup_coef_constant(void){
    const float h_ref[] = {a0,a1,a2,a3,a4};
    cudaMemcpyToSymbol(coef,h_ref,(RADIUS + 1)*sizeof(float));
}

/*
5.5.2 与只读缓存的比较
Kepler 及以后的 GPU 添加了一个功能，即使用 GPU 纹理流水线作为只读缓存，用于存储全局内存中的数据。
因为这是一个独立的只读缓存，它带有从标准全局内存读取的独立内存带宽，所以使用此功能可以为带宽限制内核带来性能优势

每个 Kepler SM 都有 48 KB 的只读缓存。一般来说，只读缓存在分散读取方面比一级缓存更好，当线程束中的线程都读取相同地址时，
不应使用只读缓存。只读缓存的粒度为32个字节。
当通过只读缓存访问全局内存时，不应使用只读缓存，需要向编译器指出在内核的持续时间里数据时只读的。
有两种方法实现这一点：
1. 使用内部函数 __ldg;
2. 全局内存的限定指针
内部函数 __ldg 用于代替标准指针解引用，并且强制加载通过只读数据缓存，如下面的代码片段所示：
output[idx] += __ldg(&input[idx]);

也可以限定指针为 const__restrict__，以表明它们应该通过只读缓存被访问：
__global__ void kernel(float *output, const float* __restrict__ input){
output[idx] += input[idx];
}

在只读缓存机制需要更多显式控制的情况下，或者在代码非常复杂以至于编译器无法检测到只读缓存的使用是否安全的情况下，
内部函数 __ldg 是一个更好的选择。

只读缓存是独立的，而且区别于常量缓存。通过常量缓存加载的数据必须是相对较小的，
而且访问必须一致以获得良好的性能（一个线程束内的所有线程在任何给定时间内应该都访问相同的位置），
而通过只读缓存加载的数据可以是比较大的，而且能够在一个非统一的模式下进行访问。
下面的内核是根据以前的模版内核修改而来的。它使用只读缓存来存储之前存储在常量内存中的系数，比较一下这两个内核，
会发现它们唯一的区别就是函数声明：
*/

// __global__ void stencil_1d_read_only(float *in,float *out,const float * __restrict__ dcoef){
//     //shared memory
//     __shared__ float smem[BDIM + 2*RADIUS];

//     //index to global memory
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;

//     //index to shared memory for stencil calculation
//     int sidx = threadIdx.x + RADIUS;

//     //Read data from global memory into shared memory
//     smem[sidx] = in[idx];

//     //read halo part to shared memory
//     if(threadIdx.x < RADIUS){
//         smem[sidx - RADIUS] = in[idx - RADIUS];
//         smem[sidx + BDIM] = in[idx+BDIM];
//     }
//     //Synchronize(ensure all the data is available)
//     __syncthreads();

//     //Apply the stencil
//     float tmp = 0.0f;
//     #pragma unroll
//     for(int i =1;i<=RADIUS;i++){
//         tmp += dcoef[i] * (smem[sidx+i] - smem[idx-i]);
//     }

//     //Store the result
//     out[idx] = tmp;
// }

__global__ void stencil_1d_read_only (float* in,
    float* out,
    const float *__restrict__ dcoef)
{
    // shared memory
    __shared__ float smem[BDIM + 2 * RADIUS];

    // index to global memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // index to shared memory for stencil calculatioin
    int sidx = threadIdx.x + RADIUS;

    // Read data from global memory into shared memory
    smem[sidx] = in[idx];

    // read halo part to shared memory
    if (threadIdx.x < RADIUS)
    {
    smem[sidx - RADIUS] = in[idx - RADIUS];
    smem[sidx + BDIM] = in[idx + BDIM];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    float tmp = 0.0f;
    #pragma unroll

    for (int i = 1; i <= RADIUS; i++)
    {
        tmp += dcoef[i] * (smem[sidx + i] - smem[sidx - i]);
    }

    // Store the result
    out[idx] = tmp;
}

int main(int argc,char **argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size
    int isize = 1 << 24;

    size_t nBytes = (isize + 2 * RADIUS) * sizeof(float);
    printf("array size: %d ", isize);

    bool iprint = 0;

    // allocate host memory
    float *h_in    = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc((float**)&d_in, nBytes));
    CHECK(cudaMalloc((float**)&d_out, nBytes));

    // initialize host array
    initialData(h_in, isize + 2 * RADIUS);

    // Copy to device
    CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

    // set up constant memory
    setup_coef_constant();

    // launch configuration
    cudaDeviceProp info;
    CHECK(cudaGetDeviceProperties(&info, 0));
    dim3 block(BDIM, 1);
    dim3 grid(info.maxGridSize[0] < isize / block.x ? info.maxGridSize[0] :
            isize / block.x, 1);
    printf("(grid, block) %d,%d \n ", grid.x, block.x);

    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS);

    


    // Copy result back to host
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));

    // apply cpu stencil
    cpu_stencil_1d(h_in, hostRef, isize);

    // check results
    checkResult(hostRef, gpuRef, isize);

    /*
    因为系数 dcoef 最初是存储在全局内存中并且读入缓存中的，调用kernel之前必须分配和初始化全局内存以便在设备上存储系数，
    代码如下所示：
    */
    const float h_coef[] = {a0,a1,a2,a3,a4};
    float *d_coef;
    cudaMalloc((float **)&d_coef,(RADIUS+1)*sizeof(float));
    cudaMemcpy(d_coef,h_coef,(RADIUS+1)*sizeof(float),cudaMemcpyHostToDevice);

    // launch read only cache kernel
    stencil_1d_read_only<<<grid, block>>>(d_in + RADIUS, d_out + RADIUS,
        d_coef);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    // // apply cpu stencil
    // cpu_stencil_1d(h_in, hostRef, isize);

    // check results
    checkResult(hostRef, gpuRef, isize);


    // print out results
    if(iprint)
    {
        printData(gpuRef, isize);
        printData(hostRef, isize);
    }

    // Cleanup
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/*
 ==25772== Profiling application: ./constantReadOnly
==25772== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   84.23%  72.311ms         2  36.156ms  10.047ms  62.264ms  [CUDA memcpy DtoH]
                   13.76%  11.813ms         3  3.9378ms  1.8880us  11.810ms  [CUDA memcpy HtoD]
                    1.00%  862.59us         1  862.59us  862.59us  862.59us  stencil_1d(float*, float*)
                    1.00%  861.82us         1  861.82us  861.82us  861.82us  stencil_1d_read_only(float*, float*, float const *)

性能近似。
由于coef 数组使用了广播访问模式，相比于只读缓存，该模式更适合于常量内存。

常量缓存于只读缓存
在设备上常量缓存和只读缓存都是只读的。
每个 SM 资源有限：常量缓存是 64 KB，只读缓存是 48KB
常量缓存在同一读取中可以更好地执行（统一读取是线程束中的每个线程都访问相同的地址）
只读缓存更适合于分散读取。
*/