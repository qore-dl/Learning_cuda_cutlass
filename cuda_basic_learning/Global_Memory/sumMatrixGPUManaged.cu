#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
/*
4.5 使用统一内存的矩阵加法
在第二章已经说明了如何在 GPU 中添加两个矩阵。
为了简化主机和设备内存空间的管理，提高这个 CUDA 程序的可读性和易维护性，
可以使用统一内存将以下解决方案添加到矩阵加法的主函数中：
1. 用托管内存分配来替换主机和设备内存分配，以消除重复指针
2. 删除所有显式的内存副本

需要在 CUDA 6.0 和 Kepler或更新的 GPU 上才可以使用内存托管

如果在一个多 GPU 设备的系统上进行测试，托管应用需要附加的步骤。
因为托管内存分配对系统中的所有设备都是可见的，所以可以限制哪一个设备对应用程序可见，这样托管的内存便可以只分配在一个设备上。
为此，设置环境变量 CUDA_VISIBLE_DEVICES 来使一个 GPU 对 CUDA 应用程序可见。

CUDA 6.0 中发布的统一内存是用来提升程序员的编程效率的。底层系统强度性能的一致性和正确性。
结果表明，在 CUDA 应用中手动优化数据移动的性能比使用统一内存的性能要更优。
可以肯定的是， NVIDIA 公司未来计划推出的硬件和软件将支持统一内存的性能提升和可编程性。
*/

void initialData(float *ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}

__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.x;

    unsigned int idx = iy*nx + ix;

    if (ix < nx && iy < ny){
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc,char **argv){
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx,ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    /*
    首先，声明和分配3个托管数组，其中数组 A 和 B 用于输入，数组gpuRef用于输出
    */
    float *A,*B,*hostRef,*gpuRef;
    CHECK(cudaMallocManaged((void **)&A,nBytes));
    CHECK(cudaMallocManaged((void **)&B,nBytes));
    CHECK(cudaMallocManaged((void **)&hostRef,nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef,nBytes));

    // initialize data at host side
    double iStart = seconds();
    //然后，使用指向托管内存的指针来初始化主机上的两个输入矩阵；
    initialData(A,nxy);
    initialData(B,nxy);
    double iElaps = seconds() - iStart;
    printf("initialization: \t %f sec\n", iElaps);
    
    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);
    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    /*
    这两种kernel 函数都需要预先执行一个warm-up函数，以避免kernel函数启动的系统开销，并获得更准确的计时结果。
    */
    sumMatrixGPU<<<grid,block>>>(A,B,gpuRef,1,1);

    // after warm-up, time with unified memory
    iStart = seconds();
    //最后，通过指向托管内存的指针调用矩阵加法核函数
    sumMatrixGPU<<<grid,block>>>(A,B,gpuRef,nx,ny);
    /*
    因为kernel函数的启动与主机程序是异步的，并且内存块 cudaMemcpy 的调用不需要使用托管内存。
    所以在直接访问 kernel 函数输出之前，需要在主机端显式地同步。
    相比于之前未托管内存的矩阵加法程序，这里的代码因为使用了统一内存而明显被简化了。
    */
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
            grid.x, grid.y, block.x, block.y);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
    
}

/*
统一内存管理结果：
$./managed 14
./managed Starting using Device 0: Tesla V100-SXM2-16GB
Matrix size: nx 16384 ny 16384
initialization:          25.113626 sec
sumMatrix on host:       1.017640 sec
sumMatrix on gpu :       0.557047 sec <<<(512,512), (32,32)>>>

不启用自动化统一内存管理：
$./manual 14
./manual Starting using Device 0: Tesla V100-SXM2-16GB
Matrix size: nx 16384 ny 16384
initialization:          13.101098 sec
sumMatrix on host:       0.967987 sec
sumMatrix on gpu :       0.004082 sec <<<(512,512), (32,32)>>>

使用统一内存管理显著地降低了性能，用nvprof跟踪两个程序：
nvprof --profile-api-trace runtime ./managed(统一内存托管)：
==104205== NVPROF is profiling process 104205, command: ./managed
./managed Starting using Device 0: Tesla V100-SXM2-16GB
Matrix size: nx 4096 ny 4096
initialization:          0.829393 sec
sumMatrix on host:       0.061483 sec
sumMatrix on gpu :       0.072581 sec <<<(128,128), (32,32)>>> 
==104205== Profiling application: ./managed
==104205== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  72.570ms         2  36.285ms  339.58us  72.230ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   73.59%  360.60ms         4  90.150ms  12.566us  360.53ms  cudaMallocManaged
                   14.81%  72.568ms         1  72.568ms  72.568ms  72.568ms  cudaDeviceSynchronize
                    9.76%  47.812ms         1  47.812ms  47.812ms  47.812ms  cudaDeviceReset
                    1.76%  8.6120ms         4  2.1530ms  1.8028ms  2.9987ms  cudaFree
                    0.07%  337.95us         1  337.95us  337.95us  337.95us  cudaGetDeviceProperties
                    0.02%  103.84us         2  51.919us  10.896us  92.943us  cudaLaunchKernel
                    0.00%  3.9600us         1  3.9600us  3.9600us  3.9600us  cudaSetDevice
                    0.00%     882ns         1     882ns     882ns     882ns  cudaGetLastError

==104205== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    7200  17.084KB  4.0000KB  0.9961MB  120.1250MB  30.48244ms  Host To Device
     382  171.39KB  4.0000KB  0.9961MB  63.93750MB  5.641770ms  Device To Host
Total CPU Page faults: 962

$sudo nvprof --profile-api-trace runtime ./manual
==104856== NVPROF is profiling process 104856, command: ./manual
./manual Starting using Device 0: Tesla V100-SXM2-16GB
Matrix size: nx 4096 ny 4096
initialization:          0.825065 sec
sumMatrix on host:       0.060143 sec
sumMatrix on gpu :       0.000383 sec <<<(128,128), (32,32)>>> 
==104856== Profiling application: ./manual
==104856== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.42%  24.319ms         2  12.159ms  12.030ms  12.289ms  [CUDA memcpy HtoD]
                   29.20%  10.229ms         1  10.229ms  10.229ms  10.229ms  [CUDA memcpy DtoH]
                    0.93%  327.04us         2  163.52us  58.528us  268.51us  sumMatrixGPU(float*, float*, float*, int, int)
                    0.45%  159.07us         2  79.535us  79.007us  80.064us  [CUDA memset]
      API calls:   51.05%  394.61ms         1  394.61ms  394.61ms  394.61ms  cudaDeviceReset
                   44.04%  340.39ms         3  113.46ms  184.05us  340.01ms  cudaMalloc
                    4.56%  35.220ms         3  11.740ms  10.423ms  12.568ms  cudaMemcpy
                    0.26%  1.9824ms         3  660.78us  204.85us  948.54us  cudaFree
                    0.04%  337.00us         1  337.00us  337.00us  337.00us  cudaDeviceSynchronize
                    0.04%  330.56us         1  330.56us  330.56us  330.56us  cudaGetDeviceProperties
                    0.01%  68.373us         2  34.186us  27.701us  40.672us  cudaLaunchKernel
                    0.01%  50.733us         2  25.366us  8.5260us  42.207us  cudaMemset
                    0.00%  3.8810us         1  3.8810us  3.8810us  3.8810us  cudaSetDevice
                    0.00%     673ns         1     673ns     673ns     673ns  cudaGetLastError

影响性能差异的最大因素在于 CPU 数据的初始化时间---使用托管内存耗费的时间更长
矩阵最初是在 GPU 上被分配的，由于矩阵是用初始值填充的，所以首先会在 CPU 上引用。
这就要求底层系统在初始化之前，将矩阵中的数据从设备传输到主机中，这是manual 版的kernel 函数中不执行的传输。

当执行主机端矩阵求和函数时，整个矩阵都在 CPU 上了，因此执行时间会相对较短。
接下来，warm-up kernel 函数将整个矩阵迁回了设备中，这样当实际的矩阵加法kernel函数
被启动时，数据已经在 GPU 上了。如果没有执行warm-up kernel 函数，使用托管内存的kernel 函数会明显地运行得更慢。

nvvp 和nvprof 支持检验统一内存的性能。这两种分析器都可以测量系统中每个 GPU 统一内存的通信量。
默认情况是不执行该功能的。通过以下的 nvprof 标志启用统一内存相关指标：
$sudo nvprof --unified-memory-profiling per-process-device ./managed
==20970== NVPROF is profiling process 20970, command: ./managed
./managed Starting using Device 0: Tesla V100-SXM2-16GB
Matrix size: nx 4096 ny 4096
initialization:          0.822689 sec
sumMatrix on host:       0.060172 sec
sumMatrix on gpu :       0.070286 sec <<<(128,128), (32,32)>>> 
==20970== Profiling application: ./managed
==20970== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  70.278ms         2  35.139ms  291.68us  69.986ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   49.83%  346.36ms         4  86.590ms  12.296us  346.30ms  cudaMallocManaged
                   38.33%  266.45ms         1  266.45ms  266.45ms  266.45ms  cudaDeviceReset
                   10.11%  70.274ms         1  70.274ms  70.274ms  70.274ms  cudaDeviceSynchronize
                    1.24%  8.6116ms         4  2.1529ms  1.7966ms  2.9759ms  cudaFree
                    0.42%  2.9356ms       808  3.6330us     142ns  192.01us  cuDeviceGetAttribute
                    0.05%  343.42us         1  343.42us  343.42us  343.42us  cudaGetDeviceProperties
                    0.01%  86.965us         2  43.482us  10.128us  76.837us  cudaLaunchKernel
                    0.01%  43.373us         8  5.4210us  3.4910us  13.334us  cuDeviceGetName
                    0.00%  24.708us         8  3.0880us  1.3560us  11.533us  cuDeviceGetPCIBusId
                    0.00%  3.5860us         8     448ns     266ns     704ns  cuDeviceTotalMem
                    0.00%  3.2680us         1  3.2680us  3.2680us  3.2680us  cudaSetDevice
                    0.00%  3.2230us        16     201ns     138ns     607ns  cuDeviceGet
                    0.00%  1.9330us         8     241ns     182ns     425ns  cuDeviceGetUuid
                    0.00%  1.1470us         3     382ns     226ns     690ns  cuDeviceGetCount
                    0.00%     394ns         1     394ns     394ns     394ns  cuModuleGetLoadingMode
                    0.00%     392ns         1     392ns     392ns     392ns  cudaGetLastError

==20970== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    6948  17.703KB  4.0000KB  876.00KB  120.1250MB  29.82539ms  Host To Device
     382  171.39KB  4.0000KB  0.9961MB  63.93750MB  5.657129ms  Device To Host
Total CPU Page faults: 962
Host-To-Device,Device-To-Host 隐式地出现大量数据传输与时间消耗（35.483ms）
与 manual 中的成本近似：
69.42%  24.319ms         2  12.159ms  12.030ms  12.289ms  [CUDA memcpy HtoD]
29.20%  10.229ms         1  10.229ms  10.229ms  10.229ms  [CUDA memcpy DtoH]
4.56%  35.220ms         3  11.740ms  10.423ms  12.568ms  cudaMemcpy

Total CPU Page faults: 962：在进行设备到主机传输数据时，将 CPU 的页故障报告给设备。
当主机应用程序引用一个 CPU 虚拟内存地址而不是物理内存地址时，就会出现页面故障。
当 CPU 需要访问当前驻留在 GPU 中的托管内存时，统一内存使用 CPU 页面故障来触发设备到主机的数据传输。
测试出的页面故障数量与数据大小密切相关。尝试用一个含有 256 x 256 个元素的矩阵重新运行程序：
$sudo nvprof --unified-memory-profiling per-process-device ./managed 8
==53795== NVPROF is profiling process 53795, command: ./managed 8
./managed Starting using Device 0: Tesla V100-SXM2-16GB
Matrix size: nx 256 ny 256
initialization:          0.003803 sec
sumMatrix on host:       0.000236 sec
sumMatrix on gpu :       0.000550 sec <<<(8,8), (32,32)>>> 
==53795== Profiling application: ./managed 8
==53795== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  546.14us         2  273.07us  238.50us  307.65us  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   83.29%  349.48ms         4  87.370ms  3.7730us  349.46ms  cudaMallocManaged
                   15.71%  65.935ms         1  65.935ms  65.935ms  65.935ms  cudaDeviceReset
                    0.70%  2.9343ms       808  3.6310us     145ns  188.52us  cuDeviceGetAttribute
                    0.13%  538.98us         1  538.98us  538.98us  538.98us  cudaDeviceSynchronize
                    0.08%  341.37us         1  341.37us  341.37us  341.37us  cudaGetDeviceProperties
                    0.05%  224.71us         4  56.177us  11.741us  158.27us  cudaFree
                    0.01%  57.779us         2  28.889us  9.4620us  48.317us  cudaLaunchKernel
                    0.01%  43.504us         8  5.4380us  3.5320us  14.121us  cuDeviceGetName
                    0.01%  25.368us         8  3.1710us  1.5210us  10.372us  cuDeviceGetPCIBusId
                    0.00%  4.0240us         1  4.0240us  4.0240us  4.0240us  cudaSetDevice
                    0.00%  3.1270us        16     195ns     133ns     598ns  cuDeviceGet
                    0.00%  3.1050us         8     388ns     287ns     573ns  cuDeviceTotalMem
                    0.00%  1.6530us         8     206ns     176ns     262ns  cuDeviceGetUuid
                    0.00%  1.1290us         3     376ns     186ns     742ns  cuDeviceGetCount
                    0.00%     615ns         1     615ns     615ns     615ns  cudaGetLastError
                    0.00%     421ns         1     421ns     421ns     421ns  cuModuleGetLoadingMode

==53795== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      29  35.310KB  4.0000KB  288.00KB  1.000000MB  201.7590us  Host To Device
      11  46.545KB  4.0000KB  128.00KB  512.0000KB  55.83900us  Device To Host
       3         -         -         -           -  537.2770us  Gpu page fault groups
Total CPU Page faults: 10
此时，页面故障数量大大减少。


*/
