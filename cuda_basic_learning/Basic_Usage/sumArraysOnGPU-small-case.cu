#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 */

/*
    验证kernel 函数代码

    除了许多可用的调试工具以外，还有两个非常简单实用的方法可以验证核函数。
    首先，可以在Fermi以及更高版本的device端的kernel函数中使用printf函数。

    其次，可以将执行参数设置为<<<1,1>>>，因此强制用一个block和一个thread执行 kernel函数，
    这模拟了串行执行程序。这对于调试和验证结果是否正确非常有用。而且，若遇到了运算次序的问题，
    这有助于你对比验证数值结果是否是按位精确的。
*/

 void checkResult(float *hostRef, float *gpuRef,const int N){
    double epsilon = 1.0E-8;
    bool match = 1;

    for(int i=0;i<N;i++){
        if(abs(hostRef[i]-gpuRef[i])>epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return; 
 }

 void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{   
    // host端：迭代N次的串行程序
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    //循环体消失，内置的线程坐标变量替换了数组索引
    //N被隐式定义用来启动N个线程
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
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
    printf("%s Starting...\n", argv[0]);

    //setup device;
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 5;
    printf("Vector size %d\n", nElem);

    //malloc host memory
    size_t nBytes = nElem*sizeof(float);

    float *h_A,*h_B,*hostRef,*gpuRef;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((float**)&d_A,nBytes));
    CHECK(cudaMalloc((float**)&d_B,nBytes));
    CHECK(cudaMalloc((float**)&d_C,nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice));

    // invoke kernel at host side
    // dim3 block (nElem);
    // dim3 grid (1);

    dim3 block (1);
    dim3 grid (nElem);

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost));
     // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return 0;
}