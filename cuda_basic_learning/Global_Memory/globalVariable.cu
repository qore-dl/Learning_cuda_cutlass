/**
全局内存

CUDA 内存模型
CUDA 内存管理
全局内存编程
探索全局内存访问模式
研究全局内存数据布局
统一内存编程
最大限度地提高全局内存吞吐量
*/

/*
4.1 CUDA 内存模型概述

内存的访问和管理是所有编程语言的重要组成部分。
在现代加速器中，内存管理对高性能计算具有重要影响

因为多数工作负载被load 和 store 数据的速度所限制，所以大量低延迟、高带宽的内存对性能是十分有利的
然而，大容量、高性能的内存造价高且不容易生产。
因此，在现有的硬件存储子系统下，必须依靠内存模型获得最佳的延迟和带宽
CUDA 内存模型结合了主机和设备的内存系统，展现了完整的内存层次结构，
使你能显式地控制数据布局以优化性能
*/

/*
4.1.1 内存层次结构的优点
一般来说，应用程序不会在某一时间点访问任意数据或运行任意代码。
应用程序往往遵循局部性原则，这表明它们可以在任意时间点访问相对较小的局部地址空间。有两种不同类型的局部性：
时间局部性
空间局部性

时间局部性认为如果一个数据位置被引用，那么该数据在较短的时间周期内很可能会再次被引用
随着时间流逝，该数据被引用的可能性逐渐降低。
空间局部性认为如果一个内存位置被引用，则附近的位置也可能被引用

现代计算机使用不断改进的低延迟低容量的内存层次结构来优化性能
这种内存层次结构仅在支持局部性原则的情况下有效。
一个内存层次结构由具有不同延迟性、带宽和容量的多级内存组成。
通常，随着从处理器到内存延迟的增加，内存的容量也在增加。一个典型的层次结构：

速度 快 -----------------------》慢
     寄存器-》缓存-》主存-》硬盘存储器
大小 最小----------------------》最大

从高到低的特点：
更低的每比特位的平均成本
更高的容量
更高的延迟
更少的处理器访问频率

CPU 和 GPU 的主存都采用的是 DRAM（动态随机存取存储器），
而低延迟内存（如 CPU L1 缓存）使用的则是 SRAM（静态随机存取存储器）
内存层次结构中最大且最慢的级别通常使用磁盘或者闪存驱动来实现。
在这种内存层次结构中，当数据被处理器频繁使用时，该数据保存在低延迟、低容量的存储器中
而当该数据被存储起来以备后用时，数据就被存储在高延迟、大容量的存储器中。
这种内存层次结构复合大内存低延迟的设想。

GPU 与 CPU 在内存层次结构设计中都使用相似的准则和模型。
GPU 和 CPU 内存模型的主要区别是，CUDA 编程模型能将内存层次结构更好地呈现给用户
能让我们显式地控制它的行为。
*/

/**
4.1.2 CUDA 内存模型
对于程序员来说，一般有两种类型的存储器：
可编程的：你需要显式地控制哪些数据存放在可编程内存中
不可编程的：你不能决定数据的存放位置，程序将自动生成存放位置已获得良好的性能

在 CPU 内存层次结构中，L1 cache 和 L2 cache都是不可编程的存储器
另一方面，CUDA 内存模型提出了多种可编程内存的类型：
寄存器
共享内存
本地内存
常量内存
纹理内存
全局内存

|(device)grid                    |
|--------------------------------|
||Block(0,0)                    ||
|||----------------------------|||
|||   共享内存                  |||
|||----------------------------|||
||        ^         ^           ||
|||-----| |         |    |-----|||
|||寄存器| |         |   |寄存器|||
|||-----| |         |    |-----||| 
||    ^   |         |      ^    ||
||    |   |         |      |    ||
||    v   v         v      v    ||
|||------------|  |------------|||
|||Thread(0,0) |  |Thread(1，0)|||                       
|||------------|  |------------|||
||    ^    ^   ^   ^    ^       ||
||    |    |   \   |    |       ||
||    v    |   |   |    v       ||
|||-------||   |   ||----------|||
|||本地内存||   |   ||本地内存  |||
|||-------||   |   ||----------|||
||---------|---|---|------------||
|          |   |   |             |
||---------v---|---|------------||   |-------|
||     全局内存 |   |            ||<->| Host  |
||-------------|---|------------||   |       | 
|              |   |             |   |  内存  |
||-------------|---|------------||   |       |
||        常量内存  |            ||<->|       |
||-----------------|------------||   |       |
|                  |             |   |       |
||-----------------|------------||   |       |
||             纹理内存          ||<->|       |
||------------------------------||   |-------|



这些内存空间的层次结构中，每种内存都有不同的作用域、生命周期和缓存行为
一个 kernel 函数中的trhead都有自己私有的本地内存
一个block 有自己的共享内存，对同一block的所有线程可见，其内容持续block的整个生命周期
所有线程都可以访问global memory。所有线程都能方位的只读内存空间为：常量内存空间、纹理内存空间
全局内存、常量内存、纹理内存（global memory、constant memory、texture memory）有不同的用途
纹理内存为各种数据布局提供了不同的寻址模式和滤波模式。
对于一个应用程序来说，全局内存、常量内存和纹理内存中的内容具有相同的生命周期。

*/

/*
4.1.2.1 寄存器
寄存器是 GPU 上运行速度最快的内存空间。
kernel 函数中声明的一个没有其他修饰符的自变量，通常存储在寄存器中。
在核函数声明的数组中，如果用于引用该数据的索引是常量且能在编译时确定，
那么该数组也存储在寄存器中。

寄存器变量对于每个线程来说都是私有的，
一个核函数通常使用寄存器来保存需要频繁访问的线程私有变量。
寄存器变量与kernel 函数的生命周期相同。
一旦kernel 函数执行完毕，就不可以对寄存器变量进行访问了

寄存器是一个在 SM 中 由活跃thread wrap 划分出的较少资源。
例如，在 Fermi GPU 中，每个线程限制最多拥有 63 个寄存器。
Kepler GPU 将限制扩展至每个thread 可以拥有 255 个寄存器。
在kernel 函数中，使用较少的寄存器将使得在 SM 上有更多的常驻block
每个 SM上并发地block 越多，其使用率和性能就通常更高。

你可以用如下的nvcc 编译器选项来检查kernel 函数使用的硬件资源的情况。
下面的命令会输出寄存器的数量、共享内存的字节数以及每个线程所使用的常量内存的字节数：
-Xptxas -v,-abi=no

如果一个kernel 函数使用了超过硬件限制数量的寄存器，
则会用local memory替代多占用的寄存器。
这种寄存器溢出会给性能带来不利影响。
nvcc 编译器使用启发式策略来最小化寄存器的使用，以避免寄存器溢出
我们也可以在代码中位每个kernel 函数显式地加上额外的信息来帮助编译器进行优化：

__global__ void __launch_bounds__(maxThreadsPerBlock,minBlokcsPerMultiprocessor)
kernel(...){

}

maxThreadsPerBlock 支出了每个block 可以包含的最大线程数
这个block 由 kernel 函数来启动。
minBlokcPerMultiProcessor 是可选参数
指明了在每个 SM 中预期的最小的常驻block 的数量
对于给定的kernel 函数，最优的启动边界会因为主要架构的版本不同而有所不同。

还可以使用 maxrregcount 编译器选项，来控制一个编译单元里所有 kernel 函数
使用的寄存器的最大数量，如：
-maxrregcount=32

若使用了指定的启动边界，则这里指定的值（32）将会失效
*/

/**
4.1.2.2 本地内存
kernel 函数中，符合存储在寄存器中，
但不能进入被该kernel 函数分配的寄存器空间中的变量将溢出到local memory 中。
编译器可能存放到 local memory 中的变量有：
1. 在编译时使用未知索引引用的本地数组
2. 可能会占用大量寄存器空间的较大本地结构体或数组
3. 任何不满足 kernel 函数寄存器限定条件的变量

“local memory" 这一名词是有歧义的；
溢出到本地内存中的变量本质上与global memory 在同一块存储区域
因此本地内存访问的特点是高延迟和低带宽，并且如在本章后面的4.3节中所描述的那样，
local memory 访问符合高效内存访问的要求。
对于计算能力2.0 及以上的 GPU 来说，本地内存数据也是存储在每个 SM 的 L1 cache 和每个 device 的L2 cache 中的。
*/

/*
4.1.2.3 共享内存
在kernel 函数处，使用如下的修饰符修饰的变量存放在shared memory 中：
__shared__

因为shared memory 是on-chip的memory，
所以与local memory 或者 全局 memory 相比，它具有更高的带宽和更低的延迟
它的使用类似于 CPU L1 Cache,但它是可编程的。

每一个 SM 都有一定数量的由block分配的shared memory。
因此必须非常小心不要过度使用shared memory，否则会在不经意间限制active wrap 的数量。

shared memory 在 kernel 函数的范围内声明，其生命周期伴随着整个block。
当一个block执行结束后，其分配的shared memory将会被释放然后重新分配给其他block。

shared memory 是thread 之间相互通信的基本方式。
一个 block 内的线程通过使用 shared memory 中的数据可以实现相互合作
访问shared memory必须同步使用如下调用，该命令在之前章节提及：
void __syncthreads();

该函数设立了一个执行障碍点，
即同一个block 中的所有线程必须在其他线程被允许执行前到达该处。
为 block 里面的所有 thread 设立障碍点，这样可以避免潜在的数据冲突。
例如，当一组未排序的多重访问通过不同的线程访问相同的内存地址时，
这些访问中至少有一个是可写的时，就会出现数据冲突。
__syncthreads 也会通过频繁强制 SM 到空闲状态来影响性能。

SM 中的 L1 cache 和 shared memory 都使用 64KB 的片上内存，它通过静态划分
但在运行时可以通过如下指令进行动态配置：
cudaError_t cudaFimcSetCacheConfig(const void* func,
                                    enum cudaFuncCache cacheConfig);

这个函数在每个 kernel 函数的基础上配置了片上内存（on-chip memory）的划分
为func 指定的 kernel 函数设置了配置。
支持的缓存配置如下：
cudaFuncCachePreferNone: 没有参考值（default)
cudaFuncCachePreferShared: 建议 48KB 的 shared memory 以及 16KB 的 L1 cache
cudaFuncCachePreferL1: 建议 48KB 的 L1 cache 和 16 KB 的 shared memory
cudaFuncCachePreferEqual: 建议相同尺寸的一级shared memory和 L1 Cache，都是32 KB
*/

/*
4.1.2.4 常量内存（constant memory）
constant memory 驻留在device memory （off-chip）中，
并且在 SM 中专用的常量缓存（constant cache）上进行缓存
常量变量使用如下修饰符来修饰：
__constant__

常量内存必须在全局空间内和所有 kernel 函数之外进行声明。
对于所有计算能力的设备，都只可以声明 64KB 的常量内存

常量内存是静态声明的，并对同一编译单元中的所有 kernel 函数可见

kernel 函数只能从 constant memory 中 load 数据。
因此，constant memory 必须在host 端使用如下的函数来初始化：

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, 
                                size_t count);
这个函数将 count 个字节从 src 指向的内存复制到symbol 指向的内存中，
这个变量存放在设备的全局内存或者常量内存中，在大多数情况下，这个函数是同步的

wrap 中的所有线程从相同的内存地址读取数据时，constant memory表现最好。
举例来说，数学公式中的系数就是一个很好的使用常量内存的例子。
因为一个wrap中的所有的线程使用相同的系数来对不同数据进行相同的计算。
如果wrap 里面的每个线程都从不同的地址空间来读取数据，
并且只读一次，那么常量内存中就不是最佳选择，
因为每次从一个常量内存中读取一次数据，都会广播给wrap里面的所有线程。
*/

/**
4.1.2.5 纹理内存
纹理内存驻留在设备内存中，并在每个 SM 的只读缓存中进行缓存。
纹理内存是一种通过指定的只读缓存访问的全局内存。
只读缓存包括硬件滤波的支持，它可以将浮点插入作为读过程的一部分来执行。
纹理内存是对二维空间局部性的优化，
所有wrap里使用纹理内存访问二维数据的线程可以达到最优性能。
对于一些应用程序来说，这是理想的内存，
并且由于缓存和滤波硬件的支持所以有较好的性能优势。
然而对于另一些应用程序来说，与global memory 相比，使用纹理内存更慢
*/

/*
4.1.2.6 全局内存
global memory 是 GPU 中最大、延迟最高且最常使用的内存。
global 指的是其作用域和生命周期。
它的生命可以在任何 SM 设备上被访问到，并且贯穿应用程序的整个生命周期

一个global memory 变量可以被静态声明或者动态声明
可以使用如下的修饰符在设备代码中静态地声明一个变量：
__device__

对于动态分配全局内存，在主机端使用cudaMalloc函数分配global memory
与此同时，在主机端使用 cudaFree 函数释放全局内存。
然后指向全局内存的指针就会作为参数传递给kernel 函数。

global memory 分配的空间在于应用程序的整个生命周期中，
并且所有 kernel 函数的所有线程可以进行访问。

从多个线程访问全局内存时需要格外注意。因为线程的执行不能跨block 同步，
不同block 内的多个线程并发地修改global memory 上的同一个位置可能会出现问题，
浙江导致一个未定义的程序行为。

global memory 常驻于 device 内存中，
可以通过 32 字节，64 字节或128字节的内存事务进行访问。
这些内存事务必须自然对齐，也就是说，首地址必须是 32 字节、64 字节或 128 字节的整数倍数

优化内存事务对于获得最优性能来说是至关重要的。当一个wrap 执行 内存 load/store时，
需要满足的传输数量通常取决于以下两个因素：
1. 跨线程的内存地址分布
2. 每个事务内存地址的对齐方式。

在一般情况下，用来满足内存请求的事务越多，
未使用的字节被传输回的可能性就越高，这造成了数据吞吐率的降低。
对于一个给定的wrap 内存请求，
事务数量和数据吞吐率是由设备的计算能力来确定的。
对于计算能力为 1.0 和 1.1 的设备，全局内存访问的要求是十分严格的。
对于计算能力高于 1.1 的设备，由于内存事务被缓存，所以要求较为宽松。
缓存的内存事务利用数据局部性来提高数据吞吐率。
接下来的部分将研究如何优化global memory access，以及如何最大程度地提高global memory的数据吞吐率
*/

/*
4.1.2.7 GPU 缓存

跟 CPU 缓存一样，GPU缓存是不可编程的内存，在 GPU 上有 4 种缓存：
一级缓存
二级缓存
只读常量缓存
只读纹理缓存

每个 SM 都有一个一级缓存，所有的 SM 共享一个 二级缓存
一级缓存和二级缓存都被用来存储
local memory 和 global memory 的数据，以及也包括寄存器溢出的部分。

对 Fermi GPU 和 Kepler K40 或者是其后发布的 GPU 来说，
CUDA 允许我们配置load 操作的数据是使用一级和二级缓存，还是只使用二级缓存。

在 CPU 上，内存的load和store都可以被缓存，
但是在 GPU 上，只有内存 load 操作可以被缓存，内存 store 操作不可以被缓存

每个 SM 也有一个只读常量cache 和只读纹理cache
它们用于在 device 内存中提高来自于各自对应内存空间内的读取性能。
*/

/*
4.1.2.8 CUDA 变量声明总结
CUDA 变量声明和它们相应的存储位置、作用域、生命周期和修饰符，总结如下：

修饰符         变量名称          存储器            作用域   生命周期
\              float var         寄存器            线程     线程
\              float var[100]   local memory      线程     线程
__shared__     float var        shared memory     block   block
__device__     float var        global memory     全局   应用程序
__constant__   float var        constant memory   全局   应有程序


接下来，我们总结各类存储器的特征：
存储器         on-chip/off-chip     缓存  存取    范围             生命周期

寄存器             on-chip          n/a   R/W    一个线程            线程
local memory      off-chip         Yes    R/W   一个线程            线程
shared memory     on-chip          n/a    R/W   块内所有线程        block
global memory     off-chip         Yes    R/W   所有线程+主机       主机配置
constant memory   off-chip         Yes    R     所有线程+主机       主机配置
texture memory    off-chip         Yes    R     所有线程+主机       主机配置
*/

/*
4.1.2.9 静态global memory
下面的代码说明了如何静态声明一个gloabl 变量。
一个浮点类型的global 变量在文件作用域内被声明。
在kernel 函数 checkGlobal-Variable中，
global 变量的值在输出之后，就发生了改变
在主函数中，global 变量的值是通过函数cudaMemcpyToSymbol完成初始化的
在执行完checkGlobalVariable 函数后，全局变量的值被替换掉了。
新的值通过cudaMemcpyFromSymbol 函数被复制回host
*/

#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable(){
    // display the original value
    printf("Device: the value of the global variable is %f\n",devData);
    //alter the value
    devData += 2.0f;
}

int main(void){
    // initialize the global variable
    float value = 3.14f;
    cudaMemcpyToSymbol(devData,&value,sizeof(float));
    printf("Host: copied %f to the global variable\n",value);

    //invoke the kernel
    checkGlobalVariable<<<1,1>>>();
    
    //copy the global variable back to the host
    cudaMemcpyFromSymbol(&value,devData,sizeof(float));
    printf("Host: the value changed by the kernel to %f\n",value);

    // 在主机端操作时，不能在 __device__ 标记的设备变量上使用运算符 &,因为它只是一个在 GPU 上表示物理位置的符号
    // 但是你可以显式地使用以下的 CUDA API 调用来获取一个全局变量的地址：
    // cudaError_t cudaGetSymbolAddress(void ** devPtr,const void *symbol);
    // 这个函数用来获取与提供设备符号相关的全局内存的物理地址，
    // 获得变量devData的地址后，可以按照如下方式使用cudaMemcpy 函数：

    float *dptr = NULL;
    cudaGetSymbolAddress((void **)&dptr,devData);
    value = 7.14;
    printf("Host: directly change the value to %f\n",value);
    cudaMemcpy(&value,dptr,sizeof(float),cudaMemcpyDeviceToHost);
    printf("Host: the value changed by the kernel to %f\n",value);

    // cudaGetSymbolAddress((void **)&dptr,devData);
    // cudaMemcpy(dptr,&value,sizeof(float),cudaMemcpyHostToDevice);


    cudaDeviceReset();
    return EXIT_SUCCESS;
}

/*
运行后结果如下：
$./globalVariable 
Host: copied 3.140000 to the global variable
Device: the value of the global variable is 3.140000
Host: the value changed by the kernel to 5.140000

尽管主机和设备的代码存储在同一个文件中，它们的执行是完全不同的
即使在同一文件内可见，host 代码也不能直接访问__device__标注的设备变量

类似地，设备代码也不能直接访问主机变量：
cudaMemcpyToSymbol(devData,&value,sizeof(float));
这一代码看似在主机上访问了设备的全局变量，但是需要注意：
1. cudaMemcpyToSymbol函数是存在在 CUDA runtime API的，即隐式地使用了 GPU 硬件来进行了访问

2. 在这里变量 devData 作为一个标识符，并不是设备全局内存中的变量地址

3. 在 kernel 函数中，devData 被当做global memory 中的一个变量

cudaMemcpy 函数不能使用如下的变量地址传递数据给devData:

cudaMemcpy(&devData,&value,sizeof(float),cudaMemcpyHostToDevice);

在主机端操作时，不能在 __device__ 标记的设备变量上使用运算符 &,
因为它只是一个在 GPU 上表示物理位置的符号
但是你可以显式地使用以下的 CUDA API 调用来获取一个全局变量的地址：

cudaError_t cudaGetSymbolAddress(void ** devPtr,const void *symbol);
这个函数用来获取与提供设备符号相关的全局内存的物理地址，
获得变量devData的地址后，可以按照如下方式使用cudaMemcpy 函数：
float *dptr = NULL;
cudaGetSymbolAddress((void **)&dptr,devData);
cudaMemcpy(dptr,&value,sizeof(float),cudaMemcpyHostToDevice);

有一个例外可以直接从主机引用 GPU 内存：CUDA 固定内存。
主机代码和设备代码都可以通过简单的指针引用直接访问固定内存。

文件作用域中的变量：可见性与可访问性

在 CUDA 编程中，你需要控制主机和设备这两个地方的操作
一般情况下，设备函数不能访问主机变量
并且主机函数也不能访问设备变量
即使这些变量在同一个文件作用域内被声明

CUDA runtime API 可以访问主机和设备变量
但是这取决于你给正确的函数是否提供了正确的参数
这样的话才能对正确的变量进行恰当的操作
因为runtime API 对某些参数的内存空间给出了假设，
如果传递了一个主机变量，而实际需要的是一个设备变量，或反之，
都可导致不可预知的后果（如应用程序崩溃）


*/