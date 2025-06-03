/*
使用CUDA runtime API 查询GPU信息
目标：查询和管理GPU设备

RUNTIME API由多个函数可以帮助管理这些设备。可以使用如下函数查询关于GPU设备的所有信息：

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop,int device);
cudaDeviceProp 结构体返回GPU设备的属性
详细可以参考：

Public Variables
int  ECCEnabled
int  accessPolicyMaxWindowSize
int  asyncEngineCount
int  canMapHostMemory
int  canUseHostPointerForRegisteredMem
int  clockRate
int  clusterLaunch
int  computeMode
int  computePreemptionSupported
int  concurrentKernels
int  concurrentManagedAccess
int  cooperativeLaunch
int  cooperativeMultiDeviceLaunch
int  deferredMappingCudaArraySupported
int  deviceOverlap
int  directManagedMemAccessFromHost
int  globalL1CacheSupported
unsigned int  gpuDirectRDMAFlushWritesOptions
int  gpuDirectRDMASupported
int  gpuDirectRDMAWritesOrdering
int  hostNativeAtomicSupported
int  hostRegisterReadOnlySupported
int  hostRegisterSupported
int  integrated
int  ipcEventSupported
int  isMultiGpuBoard
int  kernelExecTimeoutEnabled
int  l2CacheSize
int  localL1CacheSupported
char  luid[8]
unsigned int  luidDeviceNodeMask
int  major
int  managedMemory
int  maxBlocksPerMultiProcessor
int  maxGridSize[3]
int  maxSurface1D
int  maxSurface1DLayered[2]
int  maxSurface2D[2]
int  maxSurface2DLayered[3]
int  maxSurface3D[3]
int  maxSurfaceCubemap
int  maxSurfaceCubemapLayered[2]
int  maxTexture1D
int  maxTexture1DLayered[2]
int  maxTexture1DLinear
int  maxTexture1DMipmap
int  maxTexture2D[2]
int  maxTexture2DGather[2]
int  maxTexture2DLayered[3]
int  maxTexture2DLinear[3]
int  maxTexture2DMipmap[2]
int  maxTexture3D[3]
int  maxTexture3DAlt[3]
int  maxTextureCubemap
int  maxTextureCubemapLayered[2]
int  maxThreadsDim[3]
int  maxThreadsPerBlock
int  maxThreadsPerMultiProcessor
size_t  memPitch
int  memoryBusWidth
int  memoryClockRate
unsigned int  memoryPoolSupportedHandleTypes
int  memoryPoolsSupported
int  minor
int  multiGpuBoardGroupID
int  multiProcessorCount
char  name[256]
int  pageableMemoryAccess
int  pageableMemoryAccessUsesHostPageTables
int  pciBusID
int  pciDeviceID
int  pciDomainID
int  persistingL2CacheMaxSize
int  regsPerBlock
int  regsPerMultiprocessor
int  reserved[63]
size_t  reservedSharedMemPerBlock
size_t  sharedMemPerBlock
size_t  sharedMemPerBlockOptin
size_t  sharedMemPerMultiprocessor
int  singleToDoublePrecisionPerfRatio
int  sparseCudaArraySupported
int  streamPrioritiesSupported
size_t  surfaceAlignment
int  tccDriver
size_t  textureAlignment
size_t  texturePitchAlignment
int  timelineSemaphoreInteropSupported
size_t  totalConstMem
size_t  totalGlobalMem
int  unifiedAddressing
int  unifiedFunctionPointers
cudaUUID_t  uuid

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

int main(int argc,char **argv){
    printf("%s Starting...\n",argv[0]);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

     if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0;
    int driverVersion = 0;
    int runtimeVersion = 0;

    dev = 0;
    //选择最优GPU，这里选择SM数量最多的GPU
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if(numDevices>1){
        int maxMultiProcessor = 0, maxDevice = 0;
        for(int device=0;device<numDevices;device++){
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props,device);
            if (maxMultiProcessor < props.multiProcessorCount){
                maxMultiProcessor = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        dev = maxDevice;
    }

    // printf("Selected Optimal GPU: %d (name: %s)\n",dev,)
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Selected Optimal Device %d: \"%s\"\n",dev,deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf(" CUDA Driver Version / Runtime Version           %d.%d / %d.%d\n",driverVersion/1000,
            (driverVersion%100)/10,runtimeVersion/1000,(runtimeVersion%100)/10);
    printf(" CUDA Capability Major/Minor version number:     %d.%d\n",deviceProp.major,deviceProp.minor);
    printf(" Total amount of global memory:                  %.2f GBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
            (unsigned long long) deviceProp.totalGlobalMem);
    printf(" GPU Clock rate:                                 %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate * 1e-3f,deviceProp.clockRate * 1e-6f);
    printf(" Memory Clock rate:                              %.0f Mhz\n",
            deviceProp.memoryClockRate * 1e-3f);
    printf(" Memory Bus Width:                               %d-bit\n",
            deviceProp.memoryBusWidth);
    
    printf(" The number of MultiProcessor:                   %d\n",deviceProp.multiProcessorCount);
    
    if (deviceProp.l2CacheSize){
        printf(" L2 Cache Size:                                  %d bytes\n",
            deviceProp.l2CacheSize);
    }
    printf(" Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    
    printf(" Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
            deviceProp.maxTexture1DLayered[0],
            deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
    
    printf(" Total amount of constant memory:                %lu bytes\n",deviceProp.totalConstMem);
    printf(" Total amount of shared memory per block:        %lu bytes\n",deviceProp.sharedMemPerBlock);
    printf(" Total number of registers available per block:  %d\n",deviceProp.regsPerBlock);
    printf(" Warp Size:                                      %d\n", deviceProp.warpSize);
    printf(" Maximum number of threads per multiprocessor:   %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block:            %d\n",deviceProp.maxThreadsPerBlock);
    printf(" Maximum sizes of each dimension of a block:     %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf(" Maximum sizes of each dimension of a grid:      %d x %d x %d\n",deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf(" Maximum memory pitch:                           %lu bytes\n",deviceProp.memPitch);

    cudaDeviceReset();

    return 0;

}