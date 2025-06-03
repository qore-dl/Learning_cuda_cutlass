/*
4.3.4.2 示例：使用 SoA 数据布局的简单数学运算
下面kernel 函数采用 SoA 布局。分配两个一维全局内存基元数组来存储两个字段 x 和 y 的所有值。
以下 kernel 函数通过索引为每个基元数组获得取适当的值
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

struct InnerArray
{
    float x[LEN];
    float y[LEN];
};

__global__ void testInnerArray(InnerArray *data,InnerArray *result,const int n){
    unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

// functions for inner array outer struct
void initialInnerArray(InnerArray *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip->x[i] = (float)( rand() & 0xFF ) / 100.0f;
        ip->y[i] = (float)( rand() & 0xFF ) / 100.0f;
    }

    return;
}

void testInnerArrayHost(InnerArray *A, InnerArray *C, const int n)
{
    for (int idx = 0; idx < n; idx++)
    {
        C->x[idx] = A->x[idx] + 10.f;
        C->y[idx] = A->y[idx] + 20.f;
    }

    return;
}


void printfHostResult(InnerArray *C, const int n)
{
    for (int idx = 0; idx < n; idx++)
    {
        printf("printout idx %d:  x %f y %f\n", idx, C->x[idx], C->y[idx]);
    }

    return;
}

void checkInnerArray(InnerArray *hostRef, InnerArray *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon)
        {
            match = 0;
            printf("different on x %dth element: host %f gpu %f\n", i,
                   hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon)
        {
            match = 0;
            printf("different on y %dth element: host %f gpu %f\n", i,
                   hostRef->y[i], gpuRef->y[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void warmup2(InnerArray *data, InnerArray * result, const int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float tmpx = data->x[i];
        float tmpy = data->y[i];
        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

// test for array of struct
// test for array of struct
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s test struct of array at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // allocate host memory
    //用下列语句分配全局内存，注意，sizeof(InnerArray)包括其静态声明字段 x 和 y 的大小。
    int nElem = LEN;
    size_t nBytes = sizeof(InnerArray);
    
    // allocate device memory
    InnerArray *d_A, *d_C;
    CHECK(cudaMalloc((InnerArray**)&d_A, nBytes));
    CHECK(cudaMalloc((InnerArray**)&d_C, nBytes));

    InnerArray     *h_A = (InnerArray *)malloc(nBytes);
    InnerArray *hostRef = (InnerArray *)malloc(nBytes);
    InnerArray *gpuRef  = (InnerArray *)malloc(nBytes);

    // initialize host array
    initialInnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);

    

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up offset for summary
    int blocksize = 128;

    if (argc > 1) blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);

    // kernel 1:
    double iStart = seconds();
    warmup2<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup2      <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
           iElaps);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    iStart = seconds();
    testInnerArray<<<grid, block>>>(d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("innerarray   <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
           iElaps);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

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
$./simpleMethSoA 
./simpleMethSoA test struct of array at device 0: Tesla V100-SXM2-16GB 
warmup2      <<< 32768, 128 >>> elapsed 0.000190 sec
innerarray   <<< 32768, 128 >>> elapsed 0.000143 sec

$./simpleMethAoS
./simpleMethAoS test struct of array at device 0: Tesla V100-SXM2-16GB 
warmup <<<32768,128>>> elapsed 0.000210 sec
innerstruct <<<32768,128>>> elapsed 0.000152 sec

SoA 相比于 AoS 性能提升，若增大数组规模，性能提升会更明显。
通过全局加载效率和全局存储效率指标来说明这一性能提升：

SoA:Global Memory Load Efficiency     100.00%; Global Memory Store Efficiency     100.00%
$sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency ./simpleMethSoA
warmup2      <<< 32768, 128 >>> elapsed 0.036845 sec
innerarray   <<< 32768, 128 >>> elapsed 0.023176 sec

==121880== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-SXM2-16GB (0)"
    Kernel: warmup2(InnerArray*, InnerArray*, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
    Kernel: testInnerArray(InnerArray*, InnerArray*, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                            gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%

AoS:Global Memory Load Efficiency      50.00% Global Memory Store Efficiency      50.00%
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

结果说明，100%的效率说明当处理 SoA 数据布局时，load 或 store 内存请求不会重复。每次访问都由一个独立的内存事务处理
*/
