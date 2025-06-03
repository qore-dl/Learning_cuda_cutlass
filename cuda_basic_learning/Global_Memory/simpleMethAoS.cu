/*
4.3.4 结构体数组与数组结构体
C语言中两种常见的数据组织方式：数组结构体（AoS)和结构体数组（SoA）
这是一个有趣的话题，因为当存储结构化数据集时，它们代表了可以采用的两种强大的数据组织方式（结构体和数组）

下面是存储成对的浮点数据元素集的例子。首先，考虑这些成对数据元素如何使用 AoS方法进行存储，定义一个结构体，命名为innerStruct:
*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define LEN 1<<22 // 输入长度定义为 4 M；

struct innerStruct
{
    float x;
    float y;
};

/*
然后按照下面的方法定义结构体数组。这是利用 AoS 方式来组织数据的。
它存储的是空间上相邻的数据（例如,x 和 y),这在 CPU 上会有良好的缓存局部性：

struct innerStruct myAoS[N];
*/

/*
接下来，考虑使用 SoA 方法来存储数据：
*/

struct innerArray
{
    float x[LEN];
    float y[LEN];
};


/*
这里，在原结构体中每个字段的所有值都被分到各自的数组中。这不仅能将相邻数据点紧密存储起来，
也能将跨数组的独立数据点存储起来。你可以使用如下结构体定义一个变量：
struct innerArray moa;

下图说明了 AoS 和 SoA 方法的内存布局。用 AoS 模式在 GPU 上存储示例数据并执行一个只有x字段的应用程序，将导致50%的带宽损失
因为y值在每32个字节段或128个字节缓存行上隐式地被加载。 AoS 格式也在不需要的y值上浪费了二级缓存空间。

用 SoA 模式存储数据充分利用了 GPU 的内存带宽。由于没有相同字段元素的交叉存取，GPU 上的 SoA 布局提供了合并内存访问，
并且可以对全局内存实现更高效的利用。

AoS 内存布局：
               |x|y|x|y|x|y|x|y|
线程 ID：       ^    ^   ^   ^
               t0   t1  t2  t3

SoA 内存布局：
               |x|x|x|x|y|y|y|y|
线程 ID：       ^  ^ ^ ^
               t0 t1t2t3

AoS 与 SoA:
许多并行编程范式，尤其是 SIMD型范式，更倾向于使用 SoA。在 CUDA C 编程中也普遍倾向于采用 SoA。
因为数据元素是为 global memory 的有效合并访问而预先准备好的，
而被相同内存操作引用的同字段数据元素在存储时是彼此相邻的。
*/

/*
4.3.4.1 示例：使用 AoS 数据布局的简单数学运算
下述kernel 函数使用 AoS 布局。global memory 结构体数组是借助变量 x 和 y进行线性存储的。
每个线程的输入和输出是相同的：一个独立的innerStruct结构。
*/

__global__ void testInnerStruct(innerStruct *data,innerStruct *result,const int n){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

void initialInnerStruct(innerStruct *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void testInnerStructHost(innerStruct *A, innerStruct *C, const int n)
{
    for (int idx = 0; idx < n; idx++)
    {
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
    }

    return;
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                    hostRef[i].x, gpuRef[i].x);
            break;
        }

        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                    hostRef[i].y, gpuRef[i].y);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void warmup(innerStruct *data, innerStruct * result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

int main(int argc,char **argv){
    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s test struct of array at ",argv[0]);
    printf("device %d: %s \n",dev,deviceProp.name);
    cudaSetDevice(dev);

    //使用以下语句分配global memory：
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct *)malloc(nBytes);

    //initialize host array
    initialInnerStruct(h_A,nElem);
    testInnerStructHost(h_A,hostRef,nElem);

    //allocate device memory
    innerStruct *d_A,*d_C;
    cudaMalloc((innerStruct **)&d_A,nBytes);
    cudaMalloc((innerStruct **)&d_C,nBytes);

    //copy data from host to device;
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);

    //set up offset for summary
    int blocksize = 128;
    if(argc > 1){
        blocksize = atoi(argv[1]);
    }

    //execution configuration
    dim3 block(blocksize,1);
    dim3 grid ((nElem+block.x-1)/block.x,1);

    //kernel 1: warmup
    double iStart = seconds();
    warmup<<<grid,block>>>(d_A,d_C,nElem);

    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<<%3d,%3d>>> elapsed %f sec\n",grid.x,block.x,iElaps);
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef,gpuRef,nElem);

    //kernel 2: testInnerStruct
    iStart = seconds();
    testInnerStruct <<<grid,block>>> (d_A,d_C,nElem);
    cudaDeviceSynchronize();

    iElaps = seconds() - iStart;
    printf("innerstruct <<<%3d,%3d>>> elapsed %f sec\n",grid.x,block.x,iElaps);

    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef,gpuRef,nElem);

    //free memories both host and device
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    //reset device;
    cudaDeviceReset();
    return EXIT_SUCCESS;
}

/*
$./simpleMethAoS 
./simpleMethAoS test struct of array at device 0: Tesla V100-SXM2-16GB 
warmup <<<32768,128>>> elapsed 0.000216 sec
innerstruct <<<32768,128>>> elapsed 0.000154 sec

运行nvprof 命令来获取全局加载效率和全局存储效率指标：

$sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency ./simpleMethAoS
warmup <<<32768,128>>> elapsed 0.043583 sec
innerstruct <<<32768,128>>> elapsed 0.403112 sec

==72976== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: warmup(innerStruct*, innerStruct*, int)
          1                            gld_efficiency             Global Memory Load Efficiency      50.00%      50.00%      50.00%
          1                            gst_efficiency            Global Memory Store Efficiency      50.00%      50.00%      50.00%
    Kernel: testInnerStruct(innerStruct*, innerStruct*, int)
          1                            gld_efficiency             Global Memory Load Efficiency      50.00%      50.00%      50.00%
          1                            gst_efficiency            Global Memory Store Efficiency      50.00%      50.00%      50.00%


可以发现，全局内存加载效率和全局存储效率均为50%，这一结果表明，对于 AOS 数据布局，加载请求和内存存储请求是重复的。
因为字段 x 和 y 在内存中是被相邻存储的，并且有相同的大小，每当执行内存transaction时，都要加载特定字段的值
被加载的字节数的一半也必须属于其他字段。因此，请求加载和存储的 50% 带宽是未使用的。
*/



