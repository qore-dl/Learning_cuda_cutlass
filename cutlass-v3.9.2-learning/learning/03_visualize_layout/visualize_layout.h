/*! \file
  \brief CUTLASS layout visualization example
*/


#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cutlass/coord.h"
#include "cutlass/util/reference/host/tensor_foreach.h"

#include "register_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying dynamic vectors into static-length vectors 
// 允许将动态向量复制到静态长度向量
/*
在 NVIDIA 的 CUTLASS 库中，TensorCoord 是一个用于表示张量（Tensor）坐标的类。CUTLASS 是一个用于高效实现通用矩阵乘法（GEMM）和卷积运算的 CUDA 库，广泛用于深度学习和高性能计算领域。


TensorCoord 的作用
TensorCoord 的主要作用是为张量的多维坐标提供一个抽象表示。张量在计算中通常是多维的，而 TensorCoord 提供了一种简洁和统一的方式来处理这些多维坐标。


主要功能
多维坐标表示:
TensorCoord 用于表示张量在不同维度上的位置。比如，对于一个三维张量，TensorCoord 可以表示其在 x、y、z 各个维度上的位置。
封装坐标逻辑:
通过封装坐标的计算和操作逻辑，TensorCoord 提供了更高的抽象层次，使得代码更易于理解和维护。
支持运算:
TensorCoord 通常支持基本的算术运算，比如加法和减法，用于在张量空间中移动坐标。

关键点
模板化设计:
TensorCoord 通常是模板化的，以支持不同维度的张量。可以是 1D、2D、3D 或更高维度。
内存布局:
TensorCoord 与 CUTLASS 的其他组件（如张量迭代器）协同工作，以确保在 GPU 上高效地访问内存。
灵活性和可扩展性:
通过使用 TensorCoord，CUTLASS 能够以一种灵活且可扩展的方式处理不同形状和大小的张量。

通过 TensorCoord，CUTLASS 提供了一种高效且抽象的方法来处理和操作多维张量坐标，这对于实现高性能的矩阵和张量运算至关重要。

*/
template <typename TensorCoord, int Rank>
struct vector_to_coord {
  
  vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {

    coord[Rank - 1] = vec.at(Rank - 1);
    
    if (Rank > 1) {
      vector_to_coord<TensorCoord, Rank - 1>(coord, vec);
    }
  }
};

/// Permits copying dynamic vectors into static-length vectors 
template <typename TensorCoord>
struct vector_to_coord<TensorCoord,1>{
    vector_to_coord(TensorCoord &coord, std::vector<int> const &vec) {
        coord[0] = vec.at(0);
    }
};

/// Permits copying dynamic vectors into static-length vectors 
template <typename TensorCoord>
struct vector_to_coord<TensorCoord,0>{
    vector_to_coord(TensorCoord &coord,std::vector<int> const &vec){

    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> const &vec){
    auto it = vec.begin();
    if(it != vec.end()){
        out << *it;
        for(++it;it != vec.end();++it){
            out << ", " << *it;
        }
    }
    return out
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord,int Rank>
struct coord_to_vector {
    coord_to_vector(std::vector<int> &vec,TensorCoord const &coord){
        vec.at(Rank - 1) = coord[Rank - 1];
        coord_to_vector<TensorCoord,Rank - 1>(vec,coord);
    }
};

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord>
struct coord_to_vector {
    coord_to_vector(std::vector<int> &vec,TensorCoord const &coord){
        vec.at(0) = coord[0];
    }
};

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord>
struct coord_to_vector {
    coord_to_vector(std::vector<int> &vec,TensorCoord const &coord){

    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/*
在 C 语言中，inline 关键字用于建议编译器将函数扩展为内联代码，而不是通过常规的函数调用机制。这可以减少函数调用的开销，特别是在函数体较小且频繁调用的情况下。


inline 的作用
减少函数调用开销:
函数调用通常涉及压栈、跳转和返回等操作，这些操作会带来一定的性能开销。通过内联函数，编译器可以直接将函数体插入到调用点，避免这些开销。
提高代码执行效率:
对于小型、简单的函数，内联化可以显著提高执行效率，因为它消除了函数调用的开销。
建议而非命令:
需要注意的是，inline 只是对编译器的一个建议，编译器可以选择忽略这个建议。编译器会根据具体情况（如函数的复杂性和大小）决定是否进行内联

注意事项
适用场景:
inline 适用于小型、频繁调用的函数。对于大型函数，内联化可能导致代码膨胀（code bloat），影响性能。
编译器支持:
不同的编译器对 inline 的处理可能不同。通常，现代编译器会自动进行内联优化，即使没有显式使用 inline 关键字。
递归函数:
递归函数通常不适合内联化，因为内联化会导致无限递归展开。
多文件编译:
在多文件编译中，如果一个内联函数在多个源文件中使用，通常需要在头文件中定义该函数，并在所有需要使用的源文件中包含这个头文件。
C99 及以后的标准:
在 C99 及后续标准中，inline 的行为更加明确，允许在多个源文件中使用内联函数而不导致链接错误。
*/

/// Structure representing an element in source memory
struct Element {
    std::vector<int> coord; ///< logical coordinate of element (as vector)
    int offset; ///< linear offset from source memory
    int color; ///< enables coloring each element to indicate

    /// Default ctor
    inline Element(): offset(-1),color(0){ }

    /// Construct from logical coordinate and initial offset
    // 从逻辑坐标和初始偏移量构建
    
    inline Element(std::vector<int> const &coord_,int offset_,int color_ = 0):
        coord(coord_),offset(offset_),color(color_) { }
    
    /// Returns true if element is in a defined state
    inline bool valid() const {
        return offset >= 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visualizes memory layouts by constructing a 'shape' 

template<typename Layout_>
class VisualizeLayout : public VisualizeLayoutBase{
public:

    using Layout = Layout_;
    using TensorCoord = typename Layout::TensorCoord;
    using Stride = typename Layout::Stride;

public:

    Options options;
    Layout layout;
    TensorCoord extent;
    std::vector<Element> elements;

public:
    /// Initializes the problem space
    VisualizeLayout(){

    }

    /// visualization method
    bool visualize(Options const &options_){
        options = options_;

        if(options.extent.size() != TensorCoord::kRank) {
            std::cerr<< "--extent must have rank " << TensorCoord::kRank<< " (given: " << options.extent.size() << ")" << std::endl;
            return false;
        }

        vector_to_coord<TensorCoord,TensorCoord::kRank>(extent,options.extent);

        // Construct the layout for a packed tensor
        if(options.stride.empty()){
            layout = Layout::packed(extent);
        }
        else if (options.stride.size() != Stride::kRank){
            std::cerr << "--stride must have rank " << Stride::kRank << " (given: " << options.stride.size() << ")" << std::endl;
            return false;
        }
        else{
            // Stride from 
            Stride stride;
            vector_to_coord<Stride,Stride::kRank>(stride,options.stride);

            layout = Layout(stride);
        }

        // Resize elements, setting elements to 'undefined' state
        elements.resize(layout.capacity(extent));

            // enumerate points in tensor space and assign 
            /*
            template<typename Func , int Rank>
            void 	TensorForEachLambda (Coord< Rank > extent, Func func)
 	        Iterates over the index space of a tensor and calls a C++ lambda. 
            枚举张量空间中的点并赋值

            cutlass::Coord< Rank_, Index_, LongIndex_ > Struct Template Reference
            Statically-sized array specifying Coords within a tensor.

            #include <coord.h>

            Public Types
            using 	Index = Index_
 	        Index type used to store elements. More...
 
            using 	LongIndex = LongIndex_
 	        Type used to represent linear offsets.

            cutlass::Coord 是 NVIDIA 的 CUTLASS 库中用于表示多维坐标的模板类。CUTLASS 是一个高性能的 CUDA 库，主要用于实现通用矩阵乘法（GEMM）和卷积运算，而 cutlass::Coord 提供了一种通用的方式来处理多维数组或张量中的坐标。


            cutlass::Coord 的定义
            cutlass::Coord 是一个模板类，通常定义为：


            template <int Rank_>
            class Coord {
            public:
            // 类型定义
            using Index = int;

            // 成员变量
            Index idx[Rank_];

            // 构造函数
            Coord() {
                for (int i = 0; i < Rank_; ++i) {
                    idx[i] = 0;
                }
            }

            Coord(std::initializer_list<Index> list) {
                int i = 0;
                for (auto it = list.begin(); it != list.end(); ++it, ++i) {
                    idx[i] = *it;
                }
            }

            // 访问运算符
            Index &operator[](int dim) { return idx[dim]; }
            Index const &operator[](int dim) const { return idx[dim]; }

            // 其他成员函数和运算符重载
            // ...
        };

        作用
        多维坐标表示:
        cutlass::Coord 用于表示任意维度的坐标，通过模板参数 Rank_ 指定维度数。比如，Coord<3> 表示三维坐标。
        封装坐标操作:
        提供了一种统一的方式来处理多维坐标的操作，如加法、减法等。
        简化代码:
        通过封装多维坐标的逻辑，cutlass::Coord 简化了处理多维数组或张量时的代码复杂性。

        用法示例
        #include <cutlass/coord.h>
        #include <iostream>

        int main() {
            // 创建一个三维坐标
            cutlass::Coord<3> coord = {1, 2, 3};

            // 访问坐标的各个维度
            std::cout << "Coord: (" << coord[0] << ", " << coord[1] << ", " << coord[2] << ")" << std::endl;

            // 修改坐标
            coord[0] = 4;
            std::cout << "Updated Coord: (" << coord[0] << ", " << coord[1] << ", " << coord[2] << ")" << std::endl;

            // 创建另一个坐标并进行加法运算
            cutlass::Coord<3> offset = {1, 1, 1};
            cutlass::Coord<3> new_coord = coord + offset;
            std::cout << "New Coord: (" << new_coord[0] << ", " << new_coord[1] << ", " << new_coord[2] << ")" << std::endl;

            return 0;
        }

        关键点
        模板化设计:
        cutlass::Coord 是模板化的，以支持不同维度的坐标表示。可以是 1D、2D、3D 或更高维度。
        灵活性和可扩展性:
        通过使用 cutlass::Coord，程序可以灵活地处理不同维度的坐标，适用于各种高维数据结构。
        运算符重载:
        cutlass::Coord 通常支持运算符重载，如加法、减法、比较等，使得多维坐标的操作更加直观和简洁。

        cutlass::Coord 在 CUTLASS 库中提供了一种高效且抽象的方式来处理多维坐标，这对于实现复杂的张量运算是非常有用的。通过这种抽象，CUTLASS 可以更方便地处理不同形状和大小的张量。
        */
        cutlass::reference::host::TensorForEachLambda(
            extent,[&](TensorCoord coord){
                std::vector<int> coord_vec(TensorCoord::kRank,0);
                coord_to_vector<TensorCoord,TensorCoord::kRank>(coord_vec,coord);

                int offset = int(layout(coord));

                if(offset >= int(elements.size())){
                    std::cerr<< "Layout error - " << coord_vec << " is out of range (computed offset: " << offset << ", capacity: " << elements.size() << std::endl;
                    throw std::out_of_range("(TensorForEach) layout error - coordinate out of range");
                }

                elements.at(offset) = Element(coord_vec,offset);
            });
        
        return true;
    }

private:
    /// returns a pair (is_vectorizable, one_changing_rank) to determine if a vector exists (consecutive logical coordinates or uniformly invalid) at the given location. 
    //返回一个对（is_vectorizable, one_changing_rank）来确定一个向量在给定位置是否存在（连续的逻辑坐标或一致无效）。
    std::pair< bool, int > _is_vectorizable(int i) const{
        // (all elements are invalid) or 
        // // (all elements are valid AND exactly one rank is changing AND  elements are consecutive)
        //（所有元素无效）或
        // //（所有的元素都是有效的，只有一个rank在变化，元素是连续的）

        // Don't need vectorization.
        if (options.vectorize<=2){
            return std::make_pair(false,-1);
        }

        // Boundary check.
        if(i > int(elements.size()) || (i+vectorize-1) > int(element.size())){
            return std::make_pair(false,-1);
        }

        // Check if either all elements are valid or invalid.
        bool all_elements_invalid = std::all_of(
            elements.begin() + i, elements.begin() + i + options.vectorize,
            [](Element const &e) { return !e.valid(); });
        
        bool all_elements_valid = std::all_of(
            elements.begin() + i, elements.begin() + i + options.vectorize,
            [](Element const &e) { return e.valid(); });
        
        if (!all_elements_invalid && !all_elements_valid){
            return std::make_pair(false, -1);
        }
            
        // From here, it is vectorizable.
        if (all_elements_invalid) return std::make_pair(true, -1);

        // Check if only exactly one rank is changing.
        int one_changing_rank = -1;
        for(int j=0; j < options.vectorize; ++j){
            for(int r=0; r < TensorCoord::kRank; ++r){
                if(elements.at(i+j).coord.at(r) != elements.at(i).coord.at(r)) {
                    if(one_changing_rank == -1){
                        one_changing_rank = r;
                    }else if (one_changing_rank != r){
                        return std::make_pair(false,-1);
                    }
                }
            }
        }

        return std::make_pair(true,one_changing_rank);
    }

    /// Prints a vector of elements
    void _print_vector(std::ostream &out,int i, int one_changing_rank){
        Element const &base_element = elements.at(i);
        if(base_element.valid()){
            out << "(";
            for(int r=0;r < TensorCoord::kRank;++r){
                if(r){
                    out << ", ";
                }

                if(r == one_changing_rank) {
                    out<< base_element.coord.at(r)<<".."<<(base_element.coord.at(r) + options.vectorize - 1);
                }
                else{
                    out << base_element.coord.at(r);
                }
            }
            out << ")";
        }
        else{
            out << " ";
        }
    }

    /// Prints a single element
    void _print_element(std::ostream &out, int k){
        Element const &element = elements.at(k);
        if (element.valid()){
            out << "(";
            for(int v=0; v < TensorCoord::kRank;++v){
                out<<(v? ", " : "") << element.coord.at(v);
            }
            out << ")";
        }
        else{
            out <<" ";
        }
    }

public:
    /// Pretty-prints the layout to the console
    void print_csv(std::ostream &out,char delim = '|', char new_line = '\n'){
        int row = -1;

        for(int i = 0; i < int(elements.size()); i+= options.vectorize) {
            if (i % options.output_shape.at(0)) {
                out << delim;
            }
            else {
                if (row >= 0) {
                    out << new_line;
                }
                ++row;
                if (row == options.output_shape.at(1)) {
                    out << new_line;
                    row = 0;
                }
            }

            auto is_vector = _is_vectorizable(i);

            if(is_vector.first){
                _print_vector(out,i,is_vector.second); // print a vector starting at element i
            }
            else{
                for(int j=0;j < options.vectorize; ++j) { // print individual elements [i..i+j)
                    _print_element(out,i+j);
                }
            }
        }
        /*
        std::flush 是 C++ 标准库中的一个流操作符，它用于刷新输出缓冲区。具体来说，它是一个操纵器（manipulator），
        可以与输出流对象（如 std::cout）一起使用，
        以确保缓冲区的内容被立即写入到目的地（例如，终端或文件）。

        作用
        刷新输出缓冲区:
        当你向一个输出流（如 std::cout）写入数据时，数据通常会先存储在缓冲区中，然后在适当的时候（例如缓冲区满了，或者程序结束时）才实际输出到设备。std::flush 强制刷新缓冲区，将其内容立即输出。   

        确保顺序输出:
        在某些情况下，你可能希望确保输出在特定的时间点被实际写入。例如，在输出调试信息时，使用 std::flush 可以确保信息在程序崩溃之前被写出。
        */
        out << new_line << std::flush;
    }

    /// Help message
    virtual std::ostream &print_help(std::ostream &out){
        out << "TensorCoord rank " << TensorCoord::kRank << ", Stride rank: "<< Stride::kRank;
        return out;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

