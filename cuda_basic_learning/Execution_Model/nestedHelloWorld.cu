/*
动态并行

到目前为止，所有的kernel 函数都是从主机线程中被调用的。GPU 的工作负载完全在 CPU 的控制下。
CUDA 的动态并行允许在GPU端直接创建和同步新的GPU内核。
在一个kernel 函数中，在任意点动态增加 GPU 应用程序的并行性，是一个令人兴奋的新功能

到目前为止，我们需要把算法设计为单独的、大规模数据并行的内存启动。
动态并行提供了一个更有层次结构的方法，在这个方法中，并发性可以在一个GPU内核的多个level中表现出来
使用动态并行可以让递归算法更加清晰易懂，也更容易理解

有了动态并行，可以推迟到运行时决定需要在 GPU 上创建多少个 block 和 grid，
可以动态地利用 GPU 硬件调度器和加载平衡器，并进行调整以适应数据驱动或工作负载
在GPU端直接创建工作的能力可以减少在主机和设备之间传输执行控制和数据的需求，因为在设备上执行的线程可以在运行时决定启动配置

在本节中，将通过使用动态并行实现递归归约kernel 函数的例子，对如何利用动态并行做一个基本的理解
*/

/*
1. 嵌套执行
通过动态并行，我们已经熟悉了内核执行的概念（grid,block,启动配置等），也可以直接在GPU上进行kernel调用
相同的kernel 调用语法被用于在一个kernel 内部启动另一个新的kernel 函数

在动态并行中，kenrel 执行分为两种类型：父母和孩子（parent and child）
父线程、父block 或 父 grid启动一个新的grid，即子grid。
子线程、子 block 或 子 grid 被parent 启动。
子 grid 必须在父线程、父 block 或 父 grid 之前完成。
只有在所有的子 grid 都完成之后，parent 才会完成。

父grid 和子 grid 的适用范围：
host 线程配置和启动 父 grid，父grid配置和启动子 grid。
子grid的调用和完成必须进行适当地嵌套，这意味着在线程创建的所有子grid都完成之后，父grid才会完成
如果调用的thread没有显式地同步启动子网格，那么cuda runtime会保证 parent 和 child 之间的隐式同步。
显式同步例子：父 thread 可以设置栅栏，从而可以与其子 grid 显式地同步

device 线程中的 grid 启动，在 block 内是可见的。这意味着，线程可能与由该线程启动的或由相同block 中其他线程启动的子 grid 同步
在 block 中，只有当所有线程创建的所有子 grid 完成之后，block 的执行才会完成。
如果 block 中所有线程在所有的子 grid 完成之前退出，那么在那些子 grid 上隐式同步会被触发。

当parent 启动一个子 grid，父 block 与 child 显式同步之后，child才能开始执行。
父 grid 和 子 grid 共享相同的global 和 constant memory 存储
但是父 grid 和 子 grid 之间有不同的 local memory 和 shared memory
有了 child 和 parent 之间的弱一致性做保证，父 grid 和 子 grid 可以对全局内存并发load/store
有两个时刻，子 grid 和它的父 thread见到的内存完全相同：
1. 子 grid 开始时：当父thread优于子grid 调用时，所有的全局内存操作要保证对子 grid 是可见的。
2. 子 grid 结束时：当parent在子 grid 完成时进行同步操作后，子 grid 所有的内存操作应保证对parent 是可见的。

shared memory 和 local memory 分别对于block 或 thread 而言是私有的，同时，在parent 和 child 之间不是可见或一致的。
local memory 对于线程来说是私有存储，并且对该线程外部不可见
当启动一个 子 grid时，传递一个指向local memory的指针 作为参数是无效的。

*/

/*
在GPU上嵌套 Hello World
为了初步理解动态并行，可以创建一个 kernel 函数
使其动态并行输出 “Hello World!”

核函数构造的嵌套、递归执行。
host 应用程序调用 parent grid。
该 父 grid 在一个线程块中有8个线程，然后，该父 grid 中的线程 0 调用一个子grid
该子 grid 中有一半线程，即4个线程。之后，第一个grid 中的线程 0 再调用一个子 grid，
这个新的子 grid 中也只有一半线程，即2个线程

以此类推，直到最后的嵌套只有1个线程。
实现这个逻辑的kernel 代码如下所示。
每个 thread 的kernel 函数执行，会先输出 "Hello World"，接着，每个线程检查自己是否该停止。
如果在这个嵌套层里面线程数大于1，线程0就会递归地调用一个带有线程数一半的子 grid

*/

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * A simple example of nested kernel launches from the GPU. Each thread displays
 * its information when execution begins, and also diagnostics when the next
 * lowest nesting layer completes.
 */

__global__ void nestedHelloWorld(int const iSize,int iDepth){
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n",iDepth,tid,blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1){
        return;
    }

    //reduce block size to half
    int nthreads = iSize>>1;

    //thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0){
        nestedHelloWorld<<<1,nthreads>>>(nthreads,++iDepth); //非阻塞
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc,char **argv){
    int size = 8;
    int blocksize = 8; // initial block size
    int igrid = 1;

    if (argc > 1){
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block (blocksize,1);
    dim3 grid ((size+block.x-1)/block.x,1);

    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x,
           block.x);

    nestedHelloWorld<<<grid,block>>>(block.x,0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());

    return 0;

    /*
    编译方式：
    $nvcc -rdc=true nestedHelloWorld.cu -o nestedHelloWorld -lcudadevrt
    因为动态并行是由device runtime 库所支持的，所以 nestedHelloWorld 函数必须在命令行使用 -lcudadevrt 进行明确链接
    当 -rdc 标志为true 时，它强制生成可以重定位的 device code，这是动态并行的一个要求

    运行结果为：
    ./nestedHelloWorld Execution Configuration: grid 1 block 8
Recursion=0: Hello World from thread 0 block 0
Recursion=0: Hello World from thread 1 block 0
Recursion=0: Hello World from thread 2 block 0
Recursion=0: Hello World from thread 3 block 0
Recursion=0: Hello World from thread 4 block 0
Recursion=0: Hello World from thread 5 block 0
Recursion=0: Hello World from thread 6 block 0
Recursion=0: Hello World from thread 7 block 0
-------> nested execution depth: 1
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
-------> nested execution depth: 2
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0

可见，主机调用的 parent grid 有1个block，8个 thread
nestedHelloWorld kernel 函数递归调用了3次，每次调用的线程数是上次的一半
可以使用 nvvp工具证明：
nvvp ./nestHelloWorld

可视化结果：子 grid 被适当地嵌套，并且每个父 grid 会等待直到它的zi grid 执行结束
空白处说明内核在等待子 grid 执行结束

接着可以试着用两个父 block 来调用 grid 而不是一个：
./nestHelloWorld 2
输出如下：
./nestedHelloWorld Execution Configuration: grid 2 block 8
Recursion=0: Hello World from thread 0 block 0
Recursion=0: Hello World from thread 1 block 0
Recursion=0: Hello World from thread 2 block 0
Recursion=0: Hello World from thread 3 block 0
Recursion=0: Hello World from thread 4 block 0
Recursion=0: Hello World from thread 5 block 0
Recursion=0: Hello World from thread 6 block 0
Recursion=0: Hello World from thread 7 block 0
Recursion=0: Hello World from thread 0 block 1
Recursion=0: Hello World from thread 1 block 1
Recursion=0: Hello World from thread 2 block 1
Recursion=0: Hello World from thread 3 block 1
Recursion=0: Hello World from thread 4 block 1
Recursion=0: Hello World from thread 5 block 1
Recursion=0: Hello World from thread 6 block 1
Recursion=0: Hello World from thread 7 block 1
-------> nested execution depth: 1
-------> nested execution depth: 1
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
-------> nested execution depth: 2
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
-------> nested execution depth: 2
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0

我们发现，所有的子 grid 中的block 的ID都是0
这是因为子 grid 是被两个初始block 递归调用的
父grid包含两个block，所有嵌套的子grid 中仍然只包含1个block，这是因为 线程配置的kernel函数在nestedHelloWorld kernel里面嵌套实现：
//thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0){
        nestedHelloWorld<<<1,nthreads>>>(nthreads,++iDepth); //非阻塞
        printf("-------> nested execution depth: %d\n", iDepth);
    }
因此，子 grid 中的blockIdx是相对于 子 grid 而言重新编号的
    */
}

/*
动态并行的限制条件：
1. 动态并行只有在计算能力为3.5或以上的设备中才能被支持
2. 通过动态并行调用的kernel 不能在物理方面独立的device上启动。然而，在系统中允许查询任一个带 CUDA 功能的设备性能
3. 动态并行的最大嵌套深度限制为24，但是实际上，在每一个新的级别中，大多数kernel 受限于device runtime系统需要的内存size。
因为为了对每个嵌套层中的parent grid 和 child grid 之间进行同步管理，device runtime 需要保留额外的内存占用
*/