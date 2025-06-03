#include <stdio.h>
#include "common.h"
#include <cuda_runtime.h>

#define BDIMX 16
#define BDIMY 16

/*
4.4 kernel 函数可以达到的带宽

在分析 kernel 函数性能时，需要注意内存延迟，即完成一次独立内存请求的时间；
内存带宽，即 SM 访问设备内存的速度，它以每单位时间内的字节数进行测量。
在上一节中，已经尝试使用两种方法来改进kernel性能：
1. 通过最大化并行执行线程束的数量来隐藏内存延迟，通过维持更多正在执行的内存访问来达到更好的总线利用率
2. 通过适当的对齐和合并内存访问来最大化内存带宽效率

然而，当前问题的本质就是一个不好的访问模式。对于这样一个kernel函数来说，什么样的性能才是足够好的呢？
在次理想的情况下，可达到的最理想性能又是什么呢？在本节中，将利用一个矩阵转置的例子学习如何通过各种优化手段
来调整kernel函数的带宽。
即使一个原本不好的访问模式，仍然可以通过重新设计kenrel函数中的几个部分以实现良好的性能。
*/

/*
4.4.1 内存带宽
大多数kernel 函数对内存带宽非常敏感，也就是说它们有内存带宽的限制。
因此，在调整kernel 函数时需要注意内存带宽的指标。
global memory中数据的安排方式，以及线程束访问该数据的方式对带宽有显著影响。
一般有如下两种类型的带宽：
1. 理论带宽
2. 有效带宽

理论带宽是当前硬件可以实现的绝对最大带宽。对于禁用 ECC 的 Fermi M2090 来说，理论上设备内存带宽的峰值为177.6GB/s
有效带宽是kernel函数实际达到的带宽，它是测量带宽，可以用下列公式计算：

有效带宽（GB/s）= ((读字节数+写字节数)*10^(-9))/运行时间

例如，对于从设备上传入或传出数据的的拷贝来说（包含4个字节证书的 2048x2048 矩阵），有效带宽可用以下公式计算：
有效带宽（GB/s)=(2048x2048x4x2x10^{-9})/运行时间
*/

/*
4.4.2 矩阵转置问题
矩阵转置是线性代数中的一个基本问题。虽然是基本问题，但是却在许多应用程序中被使用。
矩阵的转置意味着每一列与相应的一行进行互换：
-----------         --------
|0|1| 2| 3|         |0|4|8 |
|---------|         |1|5|9 |
|4|5| 6| 7|   -》   |2|6|10|
|---------|         |3|7|11|
|8|9|10|11|         --------
-----------
*/

/*
以下是基于主机实现的使用单精度浮点值的错位转置算法。假设矩阵存储在一个一维数组中。
通过改变数组索引值来交换行和列的坐标，可以很容易得到转置矩阵：
*/

void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for( int iy = 0; iy < ny; ++iy)
    {
        for( int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

/*
在这个函数中，有两个用一维数组存储的矩阵：输入矩阵in 和转置矩阵 out。
矩阵维度被定义为nx 列 ny 行。可以用一个一维数组执行转置操作：

原始矩阵：
0 1 2 3 4 5 6 7 8 9 10 11
输出矩阵：
0 4 8 1 5 9 2 6 10 3 7 11
观察输入和输出布局，可以注意到：
读：通过原矩阵的行进行访问，结果为合并访问。
写：通过转置矩阵的列进行访问，结果为交叉访问。

交叉访问是使 GPU 性能变得最差的内存访问模式。但是，在矩阵转置操作中，这是不可避免的。
本节的剩余部分将侧重于使用两种转置kernel 函数来提高带宽的利用率：
1. 按行读取，按列存储；2. 按列读取按行存储

1. 按行读取按列存储方法：
   ix = threadIdx.x+blockIdx.x*blockDim.x
0\0------------------------------------>nx
|
|
|
|
|               (ix,iy)
|
|
|
|iy=threadIdx.y+blockIdx.y*blockDim.y
v
ny
                  |
                  V
                   
iy=threadIdx.y+blockIdx.y*blockDim.y
0\0------------------------------>ny
|
|
|
|
|               (iy,ix)
|
|
|
|
|ix = threadIdx.x+blockIdx.x*blockDim.x
|
v
nx

2. 按列读取按行存储
 ix = threadIdx.x+blockIdx.x*blockDim.x
0\0------------------------------------>nx
|
|
|
|            (iy,ix)
|              
|
|
|
|iy=threadIdx.y+blockIdx.y*blockDim.y
v
ny
                  |
                  V
                   
iy=threadIdx.y+blockIdx.y*blockDim.y
0\0------------------------------>ny
|
|
|
|
|         (ix,iy)
|
|
|
|
|ix = threadIdx.x+blockIdx.x*blockDim.x
|
v
nx
*/

/*
预测两种实现的相对性能：
如果禁用L1 Cache，那么这两种实现的性能在理论上是相同的。但是，如果启用一级缓存，那么第二种实现的性能表现会更好。
按列读取操作是不合并的（因此带宽将会浪费在未被请求的字节上），
将这些额外的字节存入一级缓存意味着下一个读操作可能会在缓存上执行而不在global memory 上执行。
因为写操作不在一级缓存中缓存，所以对按列执行写操作的例子而言，任何缓存都没有意义。在 Kepler 10、K20 和 K20x设备中，
这两种方法在性能上没有差别，因为L1 Cache 不用于global memory access。
*/

/*
4.4.2.1 为转置kernel函数设置性能的上限和下限
在执行矩阵转置kernel 函数之前，可以先创建两个拷贝kernel函数来粗略计算所有转置kernel函数的上限和下限：

通过load 和 store 行来拷贝矩阵（上限）。这样将模拟执行相同数量的内存操作作为转置，但是只通过使用合并访问。

通过load 和 store 列来拷贝矩阵（下限）。这样将模拟执行相同数量的内存操作作为转置，但是只通过使用交叉访问。

kernel 函数实现如下：
*/

// case 0 copy kernel: access data in rows
__global__ void copyRow(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny){
        out[iy*nx+ix] = in[iy*nx + ix];
    }
}

// case 1 copy kernel: access data in columns
__global__ void copyCol(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny){
        out[ix*ny + iy] = in[ix * ny + iy];
    }
}

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)( rand() & 0xFF ) / 10.0f; //100.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%dth element: %f\n", i, in[i]);
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

/*
4.4.2.2 朴素转置：读取行与读取列
基于行的朴素转置函数是基于主机实现的。这种转置按行加载，按列存储：
*/

// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if(ix < nx && iy < ny){
        out[ix * ny + iy] = in[iy*nx + ix];
    }
}

/*
通过互换读索引和写索引，就生成了基于列的朴素转置核函数。这种转置按列加载按行存储：
*/
//Case 3 transpose kernel: read in cols and write in columns
__global__ void transposeNaiveCol(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if(ix < nx && iy < ny){
        out[iy*nx+ix] = in[ix*ny+iy];
    }
}

/*
4.4.2.3 展开转置：读取行与读取列
接下来，我们将利用展开技术来提高转置内存带宽的利用率。在这个例子中，展开的目的是为每个线程分配更独立的任务，
从而最大化当前内存请求。
*/
/*
以下是一个展开因子为4的基于行的实现。这里引入了两个新的数组索引：一个用于行访问，另一个用于列访问。
*/
// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy*nx + ix; // access in rows
    unsigned int to = ix*ny + iy; // access in cols

    if (ix < nx && iy < ny){
        out[to] = in[ti];
    }

    if((ix + blockDim.x) < nx && iy < ny){
        out[to + blockDim.x * ny] = in[ti+blockDim.x];
    }

    if((ix + 2*blockDim.x) < nx && iy < ny){
        out[to + 2*blockDim.x * ny] = in[ti+2*blockDim.x];
    }

    if((ix + 3*blockDim.x) < nx && iy < ny){
        out[to + 3*blockDim.x * ny] = in[ti+3*blockDim.x];
    }
}

/*
使用相似的展开交换读索引和写索引产生一个基于列的实现：
*/
// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float *out, float *in, const int nx,
    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y + blockIdx.y + threadIdx.y;

    unsigned int ti = iy*nx + ix; // access in rows
    unsigned int to = ix*ny + iy; // access in cols 

    if (ix < nx && iy < ny){
        out[ti] = in[to];
    }

    if((ix + blockDim.x) < nx && iy < ny){
        out[ti+blockDim.x] = in[to + blockDim.x * ny];
    }

    if((ix + 2*blockDim.x) < nx && iy < ny){
        out[ti+2*blockDim.x] = in[to + 2*blockDim.x * ny];
    }

    if((ix + 3*blockDim.x) < nx && iy < ny){
        out[ti+3*blockDim.x] = in[to + 3*blockDim.x * ny];
    }

}

/*
4.4.2.4 对角转置
当启用一个线程块的网格时，线程块会被分配给 SM。编程模型抽象可能用一个一维或二维布局来表示该网格，
但是从硬件的角度来看，所有块都是一维的。每个线程块都有唯一的标识符 bid。bid可以用网格中的线程块按行优先顺序计算得出：

int bid = blockIdx.y * gridDim.x + blockIdx.x;
下面为一个 4x4 的线程块网格，包含了每个block 的 ID：
直角坐标：                             对应的块ID：
-------------------------                -------------
|(0,0)|(1,0)|(2,0)|(3,0)|                |0 |1 |2 |3 |
|-----------------------|                |-----------|
|(0,1)|(1,1)|(2,1)|(3,1)|                |4 |5 |6 |7 |
|-----------------------|                |-----------|
|(0,2)|(1,2)|(2,2)|(3,2)|                |8 |9 |10|11|
|-----------------------|                |-----------|
|(0,3)|(1,3)|(2,3)|(3,3)|                |12|13|14|15| 
------------------------|                |-----------|

当启用一个kernel函数时，block被分配给SM的顺序由block ID 来确定。一旦所有的 SM 都被完全占用，所有剩余的线程块
都保持不变直到当前的执行被完成。一旦一个线程块执行结束，将为该 SM 分配另一个线程块。由于线程块完成的速度和顺序是
不确定的，随着内核进程的执行，起初通过bid 相连的活跃线程块会变得不太连续了。
尽管无法直接调控线程块的顺序，但是可以灵活地使用block 坐标 blockIdx.x 和 blockIdx.y。例如下图展示了一个表示
blockIdx.x 和 blockIdx.y的不同方法，即使用对角块坐标系。
对角坐标：                             对应的块ID：
-------------------------                -------------
|(0,0)|(0,1)|(0,2)|(0,3)|                |0 |4 |8 |12|
|-----------------------|                |-----------|
|(1,3)|(1,0)|(1,1)|(1,2)|                |13|1 |5 |9 |
|-----------------------|                |-----------|
|(2,2)|(2,3)|(2,0)|(2,1)|                |10|14|2 |6 |
|-----------------------|                |-----------|
|(3,1)|(3,2)|(3,3)|(3,0)|                |7 |11|15|3 | 
------------------------|                |-----------|
对角坐标系用于确定一维block的 ID，但对于数据访问，仍需使用笛卡尔坐标系。
因此，当用对角坐标表示块 ID 时，需要将对角坐标系映射到笛卡尔坐标系中，以便可以访问到正确的数据块。
对于一个方阵来说，这个映射可以通过如下方程式计算而得：
block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
block_y = blockIdx.x;
这里的blockIdx.x 和 blockIdx.y为对角坐标。block_x 和 block_y 是它们对应的笛卡尔坐标。
基于行的矩阵转置kernel函数使用如下所示的对角坐标。
在kenrel 函数的起始部分包含了从对角坐标映射到笛卡尔坐标的计算。
然后使用映射到的笛卡尔坐标(block_x,block_y)来计算线程索引(ix,iy)。
这个对角转置kernel函数会影响线程块分配数据块的方式。
使用对角线程块坐标，借助合并读取和交叉写入实现了矩阵的转置
*/

/*
 * case 6 :  transpose kernel: read in rows and write in colunms + diagonal
 * coordinate transform
 */

__global__ void transposeDiagonalRow(float *out, float *in, const int nx,const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/*
基于列的对角坐标kernel函数如下所示：
*/

/*
 * case 7 :  transpose kernel: read in columns and write in row + diagonal
 * coordinate transform.
 */

__global__ void transposeDiagonalCol(float *out, float *in, const int nx,const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny){
        out[iy*nx + ix] = in[ix*ny + iy];
    }
}

int main(int argc,char **argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // select a kernel and block size
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) iKernel = atoi(argv[1]);

    if (argc > 2) blockx  = atoi(argv[2]);

    if (argc > 3) blocky  = atoi(argv[3]);

    if (argc > 4) nx  = atoi(argv[4]);

    if (argc > 5) ny  = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);

    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block(blockx,blocky);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    //allocate host memory;
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float *d_A,*d_C;

    CHECK(cudaMalloc((float **)&d_A,nBytes));
    CHECK(cudaMalloc((float **)&d_C,nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice));

    // warmup to avoide startup overhead
    double iStart = seconds();
    warmup<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup         elapsed %f sec\n", iElaps);
    CHECK(cudaGetLastError());

    // kernel pointer and descriptor
    void (*kernel)(float *,float *,int,int);
    char *kernelName;

    //set up kernel
    switch(iKernel){
        case 0:
            kernel = &copyRow;
            kernelName = "CopyRow   ";
            break;
        case 1:
            kernel = &copyCol;
            kernelName = "CopyCol   ";
            break;

        case 2:
            kernel = &transposeNaiveRow;
            kernelName = "NaiveRow      ";
            break;
    
        case 3:
            kernel = &transposeNaiveCol;
            kernelName = "NaiveCol      ";
            break;
    
        case 4:
            kernel = &transposeUnroll4Row;
            kernelName = "Unroll4Row    ";
            grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
            break;
    
        case 5:
            kernel = &transposeUnroll4Col;
            kernelName = "Unroll4Col    ";
            grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
            break;
    
        case 6:
            kernel = &transposeDiagonalRow;
            kernelName = "DiagonalRow   ";
            break;
    
        case 7:
            kernel = &transposeDiagonalCol;
            kernelName = "DiagonalCol   ";
            break;
    }

    iStart = seconds();
    kernel<<<grid,block>>>(d_C,d_A,nx,ny);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    // calculate effective_bandwidth
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;

    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> effective "
           "bandwidth %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);
    
    CHECK(cudaGetLastError());

    // check kernel results
    if(iKernel > 1){
        cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
        checkResult(hostRef, gpuRef, nx * ny, 1);
    }

    // free host and device memory
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
1. Copy 函数性能：CopyRow （上限）；CopyCol（下限）
CopyRow    elapsed 0.000067 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 500.845154 GB
CopyCol    elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 286.634399 GB

2. 朴素转置性能：
NaiveRow       elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 286.634399 G
NaiveCol       elapsed 0.000068 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 493.815735 GB

NavieCol 方法比 NavieRow 性能要好
这种性能提升的一个可能原因是在缓存行中执行了交叉读取
即使通过某一种方式读入 L1 Cache中的数据没有都被这次访问使用到，这些数据仍然留在缓存中，在以后的访问过程中可能会缓存命中
为了验证这种情况，禁用 L1 Cache 后观察kernel 函数的性能，以说明禁用 L1 Cahce load对交叉读取访问模式的影响：
CopyRow    elapsed 0.000077 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 435.719788 GB
CopyCol    elapsed 0.000145 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 231.476135 GB
NaiveRow       elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 287.219360 GB
NaiveCol       elapsed 0.000074 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 452.532104 GB

显然，按列读取的性能显著下降。按行读取略有提升。

可以通过nvprof直接观察缓存load/store性能：

copyRow(float*, float*, int, int) （按行load/store）
gld_throughput                    Global Load Throughput  347.60GB/s  347.60GB/s  347.60GB/s
gst_throughput                   Global Store Throughput  347.60GB/s  347.60GB/s  347.60GB/s

copyCol(float*, float*, int, int) (按列load/store)
gld_throughput                    Global Load Throughput  649.18GB/s  649.18GB/s  649.18GB/s
gst_throughput                   Global Store Throughput  649.18GB/s  649.18GB/s  649.18GB/s

transposeNaiveRow(float*, float*, int, int) (load 行/store 列)
gld_throughput                    Global Load Throughput  163.03GB/s  163.03GB/s  163.03GB/s
gst_throughput                   Global Store Throughput  652.14GB/s  652.14GB/s  652.14GB/s

transposeNaiveCol(float*, float*, int, int) (load 列/store 行)
gld_throughput                    Global Load Throughput  1365.9GB/s  1365.9GB/s  1365.9GB/s
gst_throughput                   Global Store Throughput  341.46GB/s  341.46GB/s  341.46GB/s

结果表明，通过缓存交叉读取能够获得最高的加载吞吐量。在缓存读取的情况下，每个内存请求由一个128字节的缓存行来完成
按列读取数据，使得wrap里的每个内存请求都会重复执行32次（因为交叉读取2048个数据元素），一旦数据预先存储到了一级缓存中，
那么许多当前全局内存读取就会有良好的隐藏延迟并取得较高的一级缓存命中率。

可以使用load/store效率来衡量性能：
copyRow(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%

copyCol(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%
gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%

transposeNaiveRow(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%

transposeNaiveCol(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%
gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%

对于 NaiveCol实现而言，由于合并写入，store 请求从未被重复执行，但是由于交叉读取，多次重复执行了load请求。
这证明了即使是较低的加载效率，L1 Cache 缓存中的缓存加载也可以限制交叉加载对性能的负面影响。

3. 展开转置性能：
CopyRow    elapsed 0.000067 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 500.845154 GB
CopyCol    elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 286.634399 GB
NaiveRow       elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 286.634399 G
NaiveCol       elapsed 0.000068 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 493.815735 G
Unroll4Row     elapsed 0.000117 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 287.219360 GB
Unroll4Col     elapsed 0.000053 sec <<< grid (32,128) block (16,16)>>> effective bandwidth 631.109802 GB
展开后，获得了一定的性能提升
尤其是按列加载和按行存储获得了更好的有效带宽和整体执行时间。

transposeNaiveRow(float*, float*, int, int) (load 行/store 列)
gld_throughput                    Global Load Throughput  163.03GB/s  163.03GB/s  163.03GB/s
gst_throughput                   Global Store Throughput  652.14GB/s  652.14GB/s  652.14GB/s
transposeUnroll4Row(float*, float*, int, int)
gld_throughput                    Global Load Throughput  163.09GB/s  163.09GB/s  163.09GB/s
gst_throughput                   Global Store Throughput  653.88GB/s  653.88GB/s  653.88GB/s

transposeNaiveCol(float*, float*, int, int) (load 列/store 行)
gld_throughput                    Global Load Throughput  1365.9GB/s  1365.9GB/s  1365.9GB/s
gst_throughput                   Global Store Throughput  341.46GB/s  341.46GB/s  341.46GB/s
transposeUnroll4Col(float*, float*, int, int)
gld_throughput                    Global Load Throughput  2172.3GB/s  2172.3GB/s  2172.3GB/s
gst_throughput                   Global Store Throughput  482.98GB/s  482.98GB/s  482.98GB/s

load/store Efficiency 不会有显著变化（可能略有下降）：
transposeNaiveRow(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%
transposeUnroll4Row(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency     100.25%     100.25%     100.25%
gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%

transposeNaiveCol(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%
gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
transposeUnroll4Col(float*, float*, int, int)
gld_efficiency             Global Memory Load Efficiency      22.23%      22.23%      22.23%
gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%

4. 使用对角坐标方式的转置：
CopyRow    elapsed 0.000067 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 500.845154 GB
CopyCol    elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 286.634399 GB
NaiveRow       elapsed 0.000117 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 286.634399 G
NaiveCol       elapsed 0.000068 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 493.815735 G
DiagonalRow    elapsed 0.000116 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 289.583313 GB
DiagonalCol    elapsed 0.000067 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 500.845154 GB

通过使用对角坐标系修改block的执行顺序（或者说是block处理数据块的顺序），这使得基于行的kernel函数性能获得了一定的提升。
展开+对角坐标可以获得更大的提升，但是实现起来较为复杂。

这种性能的提升的原因与 DRAM 的并行访问有关。发送给全局内存的请求由 DRAM 分区完成。
设备内存中连续的256字节区域被分配到连续的分区。当使用笛卡尔坐标系将线程块映射到数据块时，
全局内存访问可能无法均匀地分配到整个 DRAM 从分区中，这时候可能出现 "分区冲突"的现象。
在发生分区冲突时，内存请求在某些分区中排队等候，而另一些分区则一直未被调用。
因为对角坐标映射造成了从线程块到待处理数据块的非线性映射，
所以交叉访问不太可能落入到一个独立的分区中，并且会带来性能的提升。

对于最佳性能来说，被所有活跃的线程束并发访问的全局内存应该在分区中被均匀地划分。

假设通过两个分区访问全局内存，每个分区的宽度为256字节，并且使用一个大小为32x32的线程块启动kernel函数。
每个数据块的宽度为128字节，那么需要使用两个分区为第0个和第1个线程块加载数据。但现实是，只能使用一个分区来存储数据，造成了数据冲突。
在使用对角坐标时，需要使用两个分区为第0个和1个线程块存储和加载数据。加载和存储请求分在两个分区之间被均匀分配


*/

/*
4.4.2.5 使用瘦块来增加并行性
增加并行性最简单的方式是调整块的大小。之前的几节内容已经证明了这种简单的方法对提高性能的有效性
进一步对基于列的NaiveCol kernel 函数的block 大小进行实验：

$./transpose 3 32 32
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000156 sec
NaiveCol       elapsed 0.000078 sec <<< grid (64,64) block (32,32)>>> effective bandwidth 430.389862 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 32 16
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000149 sec
NaiveCol       elapsed 0.000075 sec <<< grid (64,128) block (32,16)>>> effective bandwidth 446.785675 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 32 8
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000156 sec
NaiveCol       elapsed 0.000073 sec <<< grid (64,256) block (32,8)>>> effective bandwidth 459.926422 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 16 32
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000154 sec
NaiveCol       elapsed 0.000066 sec <<< grid (128,64) block (16,32)>>> effective bandwidth 508.077576 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 16 16
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000145 sec
NaiveCol       elapsed 0.000064 sec <<< grid (128,128) block (16,16)>>> effective bandwidth 523.187683 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 16 8
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000158 sec
NaiveCol       elapsed 0.000078 sec <<< grid (128,256) block (16,8)>>> effective bandwidth 430.389862 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 8 32
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000149 sec
NaiveCol       elapsed 0.000065 sec <<< grid (256,64) block (8,32)>>> effective bandwidth 517.417236 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 8 16
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000157 sec
NaiveCol       elapsed 0.000076 sec <<< grid (256,128) block (8,16)>>> effective bandwidth 442.570709 GB
(base) 
[admin@speed-web011161104112.na610 /data/notebooks/cuda_learning/hq/pro-cuda-c/charpter4]
$./transpose 3 8 8
./transpose starting transpose at device 0: Tesla V100-SXM2-16GB  with matrix nx 2048 ny 2048 with kernel 3
warmup         elapsed 0.000216 sec
NaiveCol       elapsed 0.000141 sec <<< grid (256,256) block (8,8)>>> effective bandwidth 238.134491 GB

16x16 或 8x32时获得的性能较优。

但是8x32 相比 16x16 更瘦（即内层block.x维度更小）
这样一来可以增加存储操作在block中连续元素的数量，观察吞吐量和efficiency:
transposeNaiveCol(float*, float*, int, int)(16x16)
gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%
gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
gld_throughput                   Global Load Throughput  1371.6GB/s  1371.6GB/s  1371.6GB/s
gst_throughput                   Global Store Throughput  342.90GB/s  342.90GB/s  342.90GB/s

transposeNaiveCol(float*, float*, int, int)(8x32)
gld_efficiency             Global Memory Load Efficiency      50.00%      50.00%      50.00%
gst_efficiency            Global Memory Store Efficiency     100.00%     100.00%     100.00%
gld_throughput                    Global Load Throughput  693.75GB/s  693.75GB/s  693.75GB/s
gst_throughput                   Global Store Throughput  346.88GB/s  346.88GB/s  346.88GB/s

连续写的元素增多了，与此同时因为间隔读的范围减少，所以在缓存中命中的条目减少了，导致load 下降吞吐下降明显，但是间隔读的元素减少了，所以效率提升明显。


*/



