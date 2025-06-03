/*
  This example demonstrates how to use the PredicatedTileIterator in CUTLASS to load data from addressable memory,
   and then store it back into addressable memory.

  TileIterator is a core concept in CUTLASS that enables efficient loading and storing of data to and from addressable memory. 
  The PredicatedTileIterator accepts a ThreadMap type, which defines the mapping of threads to a "tile" in memory.
  This separation of concerns enables user-defined thread mappings to be specified. 

  In this example, a PredicatedTileIterator is used to load elements from a tile in global memory, stored in column-major layout, into a fragment and then back into global memory in the same layout.

  This example uses CUTLASS utilities to ease the matrix operations.

*/

/*
这个例子演示了如何使用CUTLASS中的PredicatedTileIterator从可寻址内存中加载数据，然后将其存储回可寻址内存中。

TileIterator是CUTLASS的一个核心概念，它可以有效地从可寻址内存中加载和存储数据。
PredicatedTileIterator接受ThreadMap类型，该类型定义了线程到内存中的“tile”的映射。
这种关注点分离允许指定用户定义的线程映射。

在本例中，使用PredicatedTileIterator将元素从全局内存中的tile（存储在列主布局中）加载到片段中，
然后以相同的布局返回到全局内存中。

本例使用CUTLASS util 程序简化矩阵操作。
*/


// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

//
//  CUTLASS utility includes
//

//Defines operator<<() to write TensorView Objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"


// Defines cutlass::reference::host::TensorFill() and
// cutlass::reference::host::TensorFillBlockSequential()
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

// #pragma warning(disable : 4503)

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define PredicatedTileIterators to load and store a M-by-K tile, in column major layout.
// 定义PredicatedTileIterators以列主布局加载和存储m × k的tile。

template<typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<2> extent){
    
    Iterator dst_iterator(dst_params,dst_pointer,extent,threadIdx.x);
    Iterator src_iterator(src_params,src_pointer,extent,threadIdx.x);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided

    /*
    PredicatedTileIterator使用PitchLinear布局，因此采用PitchLinearShape。
    可以通过 `Iterator::Shape::kContiguous` 访问连续维度，通过 `Iterator::Shape::kStrided` 访问步长维度。
    */
    int iterations = (extent[1]+Iterator::Shape::kStrided-1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment fragment;

    for(int i=0;i<fragment.size();++i){
        fragment[i] = 0;
    }

    src_iterator.load(fragment);
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("\nAll threads dump all the elements:\n");
    }
    /*
    frag: frag 是一个片段对象，通常代表一个小块的矩阵或张量数据。片段在 Cutlass 中用于在 GPU 上执行分块计算，以便更好地利用硬件资源。
    cutlass::debug::dump_fragment:这是一个调试函数，用于将片段 frag 的内容打印到标准输出或日志中。
    它可以帮助开发者在调试和验证算法时，检查片段中的数据是否正确，或者是否符合预期。
    这种调试输出对于理解和优化复杂的 GPU 内核计算非常有用，因为它允许开发者查看计算过程中每个步骤的中间结果。
    在使用这个函数时，需要确保编译时启用了调试支持，因为在某些情况下，调试功能可能在发布版本中被禁用以优化性能。
    通过这种调试工具，开发者可以更好地理解程序的行为，尤其是在开发和优化 GPU 加速的数值计算时。
    */
    
    dst_iterator.store(fragment);
    cutlass::debug::dump_fragment(fragment);

    ++src_iterator;
    ++dst_iterator;

    for(;iterations>1;--iterations){
        src_iterator.load(fragment);
        dst_iterator.store(fragment);
        cutlass::debug::dump_fragment(fragment);

        ++src_iterator;
        ++dst_iterator;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes the source tile with sequentially increasing values and performs the copy into the destination tile using two PredicatedTileIterators,
// one to load the data from addressable memory into a fragment (regiser-backed array of elements owned by each thread) 
// and another to store the data from the fragment back into the addressable memory of the destination tile.

/*
//用顺序递增的值初始化源tile，并使用两个predicatedtileiterator将拷贝到目标tile中；
//一个从可寻址内存加载数据到一个片段（每个线程拥有的寄存器支持的元素数组）
//另一个将片段中的数据存储回目标tile的可寻址内存。
*/

cudaError_t TestTileIterator(int M,int K){
    // For this example, we chose a <64, 4> tile shape. The PredicatedTileIterator expects
    // PitchLinearShape and PitchLinear layout.
    using Shape = cutlass::layout::PitchLinearShape<64,4>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = int;
    int const kThreads = 32;
   
    // ThreadMaps define how threads are mapped to a given tile. 
    // The PitchLinearStripminedThreadMap stripmines a pitch-linear tile among a given number of threads, 
    // first along the contiguous dimension then along the strided dimension.

    /*
    ThreadMap 定义了线程如何映射到给定的瓦片。
    PitchLinearStripminedThreadMap 在给定数量的线程中对 pitch-linear 瓦片进行条带化(stripmined)，
    首先沿着连续维度，然后沿着步长维度。
    */
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape,kThreads*2>;
    /*
    使用前kThreads*2个线程，协同读取一个64x4的tile
    */
    // Define the PredicatedTileIterator, using TileShape, Element, Layout, and ThreadMap types
    /*
    定义tile 迭代器，使用 PredicatedTileIterator 类型，
    需要使用的要素包括：
        TileShape (Tile 的大小)
        Element： 元素的类型
        Layout 是矩阵的存储布局，比如行优先或列优先。
        ThreadMap：Warp内线程：Tile 和 线程的映射方案   
    */
    using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
        Shape,Element,Layout,1,ThreadMap>; ///*advanced rank=1?*/

    cutlass::Coord<2> copy_extent = cutlass::make_Coord(M,K);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(M,K);

    // Allocate source and destination tensors
    /*
    分配源和目标 tensor 空间
    */
    cutlass::HostTensor<Element,Layout> src_tensor(alloc_extent);
    cutlass::HostTensor<Element,Layout> dst_tensor(alloc_extent);

    Element oob_value = Element(-1);

    // Initialize destination tensor with all -1s
    cutlass::reference::host::TensorFill(dst_tensor.host_view(),oob_value);
    // Initialize source tensor with sequentially increasing values
    cutlass::reference::host::BlockFillSequential(src_tensor.host_data(),src_tensor.capacity());
    // cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

    dst_tensor.sync_device();
    src_tensor.sync_device();

    // Dump the Matrix:
    std::cout << "Matrix:\n" << src_tensor.host_view() << "\n";

    typename Iterator::Params dst_params(dst_tensor.layout());
    typename Iterator::Params src_params(src_tensor.layout());

    dim3 block(kThreads*2,1);
    dim3 grid(1,1);

    // Launch copy kernel to perform the copy
    copy<Iterator><<<grid,block>>>(
        dst_params,
        dst_tensor.device_data(),src_params,src_tensor.device_data(),copy_extent);
    
    cudaError_t result = cudaGetLastError();
    if(result != cudaSuccess) {
        std::cerr << "Error - kernel failed." << std::endl;
        return result;
    }

    dst_tensor.sync_host();

    // Verify results
    for(int s = 0; s < alloc_extent[1]; ++s) {
        for(int c = 0; c < alloc_extent[0]; ++c) {
            Element expected = Element(0);
            if(c < copy_extent[0] && s < copy_extent[1]) {
                expected = src_tensor.at({c, s});
            }
            else {
                expected = oob_value;
            }
            Element got = dst_tensor.at({c, s});
            bool equal = (expected == got);
  
            if(!equal) {
                std::cerr << "Error - source tile differs from destination tile." << std::endl;
                return cudaErrorUnknown;
            }
        }
    }
  
    return cudaSuccess;
}

int main(int argc, const char *arg[]) {

    cudaError_t result = TestTileIterator(57, 35);

    if(result == cudaSuccess) {
      std::cout << "Passed." << std::endl;  
    }

    // Exit
    return result == cudaSuccess ? 0 : -1;
}