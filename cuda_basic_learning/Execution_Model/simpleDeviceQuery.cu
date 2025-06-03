/*
资源分配

线程束的本地执行上下文主要由以下资源组成：
程序计数器
寄存器
共享内存

由 SM 处理的每个线程束的执行上下文，在整个线程束的生命周期中是保存在芯片内的。
因此，从一个执行上下文切换到另一个执行上下文没有损失。

每个 SM 都有 32 位的寄存器组，它存储在寄存器文件中，并且可以在线程中进行分配
同时固定数量的共享内存用来在block中进行分配，
对于一个给定的kernel，同时存在于同一个 SM 中的 block 和 thread warp 的数量
取决于在SM中可用的且内核所需的寄存器和共享内存的size

定性来看，若每个线程消耗的寄存器越多，则可以放在一个 SM 中的线程束就越少。
若可以减少 kernel 消耗寄存器的数量，那么就可以同时处理更多的thread warp。

若一个block消耗的shared memory 越多，则在一个 SM 中可以被同时处理的block 就会变少
若每个block使用的shared memory 变少，则可以同时处理更多的block

资源可用性通常会限制 SM 中常驻的block的数量，
每个 SM 中寄存器和 shared memory 的数量因设备拥有不同的计算能力而不同。而若每个 SM 中没有足够的寄存器或shared memory
去处理至少一个 block，那么内核将无法启动。

当计算资源（如 register 和 shared memory）已分配给block 时， block 被称为 活跃的block。
该block 包含的thread wrap 被称为活跃的 thread wrap，活跃的线程束可以进一步被分为以下三种类型：
1. 选定的线程束
2. 阻塞的线程束
3. 符合条件的线程束

一个 SM 上的 warp scheduler 在每个周期都选择活跃的线程束，然后把他们调度到执行单元上执行。
活跃执行的线程束被称为选定的线程束。如果一个活跃的线程束准备执行但尚未执行，则被称为一个符合条件的线程束
若一个线程束没有做好执行的准备，它是一个阻塞的线程束。如果同时满足以下两个条件，则线程束符合执行条件：
1）32 个 CUDA 核心可用于执行；
2）当前指令中的所有参数都已就绪

例如，Kepler SM上活跃的线程束数量，从启动到完成在任何时候都必须小于或等于 64 个并发线程束的架构限度。
在任何时期，选定的线程束的数量都小于或等于 4。若在执行中，线程束阻塞，线程束调度器会令一个符合条件的线程束代替它去执行。
由于计算资源是在thread warp之间分配的，而且线程束的整个生命周期中都保持在芯片中，因此线程束上下文切换是非常快的

为了隐藏由线程阻塞造成的延迟，需要让大量的线程束保持活跃

在 CUDA 编程中需要特别关注计算资源分配：计算资源限制了活跃的线程束的数量。因此必须要了解由硬件产生的限制和kenrel用到的资源。
为了最大程度地利用 GPU，需要最大化活跃的thread wrap的数量。

延迟隐藏

SM 依赖线程级的并行，以最大化功能单元的利用率。
因此，利用率与常驻线程束的数量直接相关。
在指令发出和完成之间的时钟周期被定义为指令延迟。
当每个时钟周期中所有的wrap scheduler 都有一个符合条件的线程束时，可以达到计算资源的完全利用。
这就可以保证，通过在其他常驻线程束中发布其他指令，可以隐藏每个指令的延迟。

与在 CPU 上用 C 语言编程相比，延迟隐藏在 CUDA 编程中尤为重要。
CPU 核心是为同时最小化一个或两个线程的延迟而设计的，而 GPU则是为处理大量并发和轻量级线程以最大化吞吐量而设计的
GPU的指令延迟被其他线程束的计算隐藏。

考虑到指令延迟，指令可以被分为两种基本类型：
算术指令
内存指令

算术指令延迟是一个算术操作从开始到它产生输出之间的时间。
内存指令延迟是指发送出的load或store操作和数据到达目的地之间的时间。
对于每种情况，相应的延迟大约为：

算术操作约为 10~20个周期
global memory access 约为 400~800个周期

如何估算隐藏延迟所需要的活跃线程束的数量：
Little's Law 可以提供一个合理的近似值。起源于队列理论中的一个定理，可以应用于 GPU中：
所需线程束数量 = 延迟 x 吞吐量

吞吐量和带宽

带宽和吞吐量经常被混淆，根据实际情况它们可以被交换使用。吞吐量和带宽都是度量性能的速度指标

带宽通常是指理论峰值，而吞吐量是指已达到的值
带宽通常是用来描述单位时间内最大可能得数据传输量，
而吞吐量是用来描述单位时间内任何形式的信息或操作的执行速度，例如每个周期完成了多少个指令。

对于算术运算来说，其所需的并行可以表示成隐藏算术延迟所需要的操作数量。
（常见标准操作：32位浮点数的乘加运算，吞吐量：在每个SM中每个周期内的操作数量）
吞吐量因为不同的算术指令而不同

吞吐量由SM中每个周期的操作数量来确定，而执行一条指令的一个线程束对应32个操作。
因此为了保持计算资源的充分利用，对于 Fermi GPU 而言，每个 SM 中所需的线程束数量通过计算：640/32=20个线程束
因此，算术运算所需的并行可以用操作的数量或者线程束的数量来表示。这个简单的单位转换说明，可以用两种方法提高并行：
1. 指令级并行 （ILP）：一个线程拥有很多独立的指令
2. 线程级并行 （TLP）：很多并发地符合条件的线程

对于内存操作来说，其所需的并行可以表示为每个周期内隐藏内存延迟所需的字节数
因为 内存吞吐量通常表示为每秒千兆字节数，
所以首先需要用对应的内存频率将吞吐量转换为每周期千兆字节数，可以使用以下的命令检测设备的内存频率：
nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"

1Hz 定义为每秒1个周期，因此可以把带块从每秒千兆字节数转换为每周期千兆字节数，公式如下：
144GB/s （吞吐量/内存带宽）/ 1.566GHz (内存访问时钟频率) = 92 字节/周期

用内存延迟x每周期字节数，可以获得 Fermi 内存操作所需的并行：约74KB的内存 I/O运行，用以实现充分的利用

因为内存带宽是对于整个设备而言的，因此这个值是对于整个设备，而不是对于每个 SM来说的

利用应用程序，把这些值与线程束或线程数量关联起来，
假设每个线程都把一个浮点数据（4byte）从global memory 移动到 SM 中用于计算，
则在 Fermi GPU 上，总共需要 18 500个线程或者 579个线程束来隐藏所有内存延迟，具体计算为：
74KB/4(74000/4) = 18500 个线程
18500个线程 / 32个线程每线程束 = 579 个线程束

Fermi 架构有16个 SM，因此579个线程束/16个SM = 36 线程束/SM，以隐藏所有的内存延迟。
如果每个线程执行多个独立的4字节加载，隐藏内存延迟需要的线程就可以更少

与指令延迟很像，通过在每个线程/线程束中创建更多独立的内存操作，
或创建更多并发地活跃的线程/线程束，可以增加可用的并行。

延迟隐藏取决于每个 SM 中活跃线程的数量，这一数量由执行配置和资源约束隐式决定（一个kernel中寄存器和shared memory的使用情况）
选择一个最优执行配置的关键是在延迟隐藏和资源利用之间达到一个平衡。

显式充足的并行
因为GPU在线程间分配计算资源并在并发线程束之间切换的消耗（在一个或两个周期命令上）很小，所以所需的状态可以在芯片内获得
如果有足够的并发活跃线程，那么可以让 GPU 在每个时钟周期内的每个流水线阶段中忙碌。在这种情况下，一个线程束的延迟可以被
其他线程束的执行所隐藏。因此，向 SM显示足够的并行对性能是有利的。

计算所需并行的一个简单公式是，用每个 SM 核心的数量乘以在该 SM 上一条算术指令的延迟。
例如，Fermi有32个单精度浮点流水线线路，一个算术指令的延迟是20个周期，
所以每个SM至少需要有32x20=640个线程使得设备处于忙碌状态，然而，这只是一个下边界。

*/

/*
占用率（occupy）
在每个 CUDA 核心里，指令是顺序执行的。当一个线程束阻塞时，SM切换执行其他符合条件的线程束。
在理想情况下，我们想要有足够的线程束占用设备的核心。占用率是每个 SM 中活跃的线程束占最大线程束数量的比值：
占用率 = 活跃线程束数量 / 最带线程束数量
使用以下函数，可以检测设备中每个 SM 的最大线程束数量：
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,int device)

来自设备的各种统计数据在cudaDeviceProp结构中被返回。每个 SM 中线程数量的最大值在以下变量中返回：

    maxThreadsPerMultiProcessor
*/

#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * Fetches basic information on the first device in the current CUDA platform,
 * including number of SMs, bytes of constant memory, bytes of shared memory per
 * block, etc.
*/

int main(int argc,char *argv[]){
    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp,iDev);

    printf("Device %d: %s\n",iDev,iProp.name);
    printf("Number of multiprocessors:                        %d\n",iProp.multiProcessorCount);
    printf("Total amount of constant memory:                  %4.2f KB\n",iProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block:          %4.2f KB\n",iProp.sharedMemPerBlock/1024.0);
    // printf("Total amount of shared memory per multiprocessor: %4.2f KB\n",iProp.sharedMemPerMultiProcessor/1024.0);
    printf("Total number of registers available per block:    %d\n",iProp.regsPerBlock);
    printf("Warp size:                                        %d\n",iProp.warpSize);
    printf("Maximum number of threads per block:              %d\n",iProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor:     %d\n",iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor:       %d\n",iProp.maxThreadsPerMultiProcessor/32);
    return EXIT_SUCCESS;
}

/*
占用率计算器包含几个部分，首先，必须提供 GPU 的计算能力和kernel的资源使用情况的信息

在确定 GPU 的计算能力后，物理限制部分的数据是自动填充的。接下来，需要输入以下kernel 资源信息：
每个block的线程（执行配置）
每个block的寄存器 （资源使用情况）
每个block的共享内存（资源使用情况）

每个 thread 的寄存器和每个 block 的共享内存资源的使用情况可以从 nvcc 中用以下编译器标志获得：
--ptxas-options=-v
一旦进入这个数据，kernel占用率便会展示在 GPU 占用率数据段。
其他部分提供必要的信息，来调整执行配置和资源使用情况，已获得更好的设备占用率
kernrel 使用的寄存器数量会对常驻线程束数量产生显著的影响，寄存器的使用可以使用下面的
nvcc标志手动控制：
-maxrrgecount=NUM

-maxrregcount 选项告诉编译器每个线程使用的寄存器数量不超过 NUM 个。使用这个编译器标志，
可以得到占用率计算器推荐的寄存器数量，同时使用这个数值可以改善应用程序的性能。

为了提高占用率，还需要调整 block 的配置或者重新调整资源的使用情况，
以允许更多的线程束同时处于活跃状态和提高计算资源的利用率。

极端地操纵block会限制资源的利用：
小 block：每个 block 中 thread 太少，会在所有资源被充分利用之前导致硬件达到每个 SM 的线程束数量的限制。
大 block：每个 block 中有太多的线程，会导致在每个 SM 中每个线程可用的硬件资源较少

grid 和 block 大小的限制：
使用这些准则可以使应用程序适用于当前和将来的device：
1. 保持每个block中 thread 数量是 线程束 warp 大小(32)的倍数
2. 避免 block 太小：每个 block 至少要有 128 或 256 个线程
3. 根据 kernel 资源的需求调整 block 大小
4. block 的数量要远远多于 SM 的数量，从而在设备中可以显示有足够的并行。
5. 通过实验得到最佳执行配置和资源使用情况

尽管在每种情况下会遇到不同的硬件限制，但它们都会导致计算资源未充分利用。
阻碍隐藏指令和内存延迟的并行的建立。
占用率唯一注重的是在每个 SM 中 并发线程或线程束的数量。然而，充分的占用率不是性能优化的唯一目标
kernel 一旦达到一定级别的占用率，进一步增加占用率可能不会改进性能。为了提高性能，可以调整很多其他因素

*/