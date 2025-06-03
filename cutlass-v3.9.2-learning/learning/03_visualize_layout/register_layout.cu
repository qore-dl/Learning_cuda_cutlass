/*! \file
  \brief CUTLASS layout visualization example
*/

#include <map>
#include <memory>

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/patch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "visualize_layout.h"
#include "register_layout.h"

/*
std::unique_ptr 是 C++11 引入的一种智能指针类型，位于 <memory> 头文件中。它用于管理动态分配的对象，
确保对象在不再需要时自动释放内存，从而避免内存泄漏。

特性
独占所有权:
std::unique_ptr 拥有其所管理对象的独占所有权。不能复制一个 std::unique_ptr，只能通过移动语义转移所有权。
自动释放:
当 std::unique_ptr 超出作用域时，它会自动销毁其所管理的对象并释放内存。
轻量级:
std::unique_ptr 是一个轻量级的智能指针，开销小于 std::shared_ptr，因为它不需要引用计数。

用法示例
#include <iostream>
#include <memory>

class MyClass {
public:
    MyClass() { std::cout << "MyClass constructed\n"; }
    ~MyClass() { std::cout << "MyClass destructed\n"; }
    void sayHello() { std::cout << "Hello from MyClass\n"; }
};

int main() {
    // 创建一个 std::unique_ptr 管理 MyClass 对象
    std::unique_ptr<MyClass> ptr1(new MyClass());

    // 使用 make_unique 创建 unique_ptr (C++14 引入)
    auto ptr2 = std::make_unique<MyClass>();

    // 使用 unique_ptr
    ptr1->sayHello();
    ptr2->sayHello();

    // 转移所有权
    std::unique_ptr<MyClass> ptr3 = std::move(ptr1);
    if (!ptr1) {
        std::cout << "ptr1 is now empty\n";
    }

    // ptr3 仍然有效
    ptr3->sayHello();

    // 当 ptr3 超出作用域时，MyClass 对象会自动销毁
    return 0;
}

关键点
不能复制:
std::unique_ptr 不支持复制构造或赋值。这是因为它的设计就是为了保证独占所有权。要转移所有权，可以使用 std::move。
自定义删除器:
可以提供自定义删除器来指定如何销毁对象。例如，使用函数对象或 lambda 表达式。
std::unique_ptr<MyClass, void(*)(MyClass*)> ptr(new MyClass(), [](MyClass* p) {
    std::cout << "Custom deleter called\n";
    delete p;
});
数组支持:
std::unique_ptr 可以管理动态数组，使用时需要指定数组类型。例如，std::unique_ptr<int[]> arr(new int[10]);。
与原生指针兼容:
可以通过 get() 方法获取底层原生指针，但应谨慎使用，避免手动管理内存导致的问题。

std::unique_ptr 是管理动态内存的一个强大工具，通过它可以确保资源的安全释放，减少内存泄漏的风险。它的独占所有权模型使得代码更加清晰和安全。
*/

/////////////////////////////////////////////////////////////////////////////////////////////////
void RegisterLayouts(std::map<std::string,std::unique_ptr<VisualizeLayoutBase>> &layouts) {
    struct{
        char const *name;
        VisualizeLayoutBase *ptr;
    } layout_pairs[] = {
        {"PitchLinear", new VisualizeLayout<cutlass::layout::PitchLinear>},
        {"ColumnMajor", new VisualizeLayout<cutlass::layout::ColumnMajor>},
        {"RowMajor", new VisualizeLayout<cutlass::layout::RowMajor>},
        {"ColumnMajorInterleaved<4>",
         new VisualizeLayout<cutlass::layout::ColumnMajorInterleaved<4>>},
        {"RowMajorInterleaved<4>",
         new VisualizeLayout<cutlass::layout::RowMajorInterleaved<4>>},
        // All Ampere/Turing H/Integer matrix multiply tensor core kernels uses the same swizzling
        // layout implementation with different templates.
        //
        // mma.sync.aligned.m8n8k128.s32.b1.b1.s32 Interleaved-256
        // mma.sync.aligned.m16n8k256.s32.b1.b1.s32 Interleaved-256
        {"TensorOpMultiplicand<1,256>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<1, 256>>},
        // mma.sync.aligned.m8n8k128.s32.b1.b1.s32 TN kblock512
        // mma.sync.aligned.m16n8k256.s32.b1.b1.s32 TN kblock512
        {"TensorOpMultiplicand<1,512>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<1, 512>>},
        // mma.sync.aligned.m16n8k256.s32.b1.b1.s32 TN kblock1024
        {"TensorOpMultiplicand<1,1024>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<1, 1024>>},
        // Integer matrix multiply.int4 8832  Interleaved-64
        // Integer matrix multiply.int4 16864 Interleaved-64
        {"TensorOpMultiplicand<4,64>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<4, 64>>},
        // Integer matrix multiply.int4 8832  TN kblock128
        // Integer matrix multiply.int4 16864 TN kblock128
        {"TensorOpMultiplicand<4,128>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<4, 128>>},
        // Integer matrix multiply.int4 16864 TN kblock256
        {"TensorOpMultiplicand<4,256>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<4, 256>>},
        // Integer matrix multiply 8816  Interleaved-32
        // Integer matrix multiply 16832 Interleaved-32
        {"TensorOpMultiplicand<8,32>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<8, 32>>},
        // Integer matrix multiply 8816  TN kblock64
        // Integer matrix multiply 16832 TN kblock64
        {"TensorOpMultiplicand<8,64>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<8, 64>>},
        // Integer matrix multiply 16832 TN kblock128
        {"TensorOpMultiplicand<8,128>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<8, 128>>},
        // Matrix Multiply 1688  TN kblock32
        // Matrix multiply 16816 TN kblock32
        {"TensorOpMultiplicand<16,32>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<16, 32>>},
        // Matrix multiply 1688  NT
        // Matrix multiply 16816 NT
        // Matrix multiply 16816 TN kblock64
        {"TensorOpMultiplicand<16,64>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<16, 64>>},
        // Matrix multiply 1688.TF32 TN kblock16
        {"TensorOpMultiplicand<32,16>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<32, 16>>},
        // Matrix multiply 1688.TF32 TN kblock32
        {"TensorOpMultiplicand<32,32>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<32, 32>>},
        // Matrix multiply 1688 NT
        {"TensorOpMultiplicandCongruous<32,32>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCongruous<32, 32>>},
        // Matrix multiply 884 NT
        {"TensorOpMultiplicandCongruous<64,16>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCongruous<64, 16>>},
        // Matrix multiply 884 TN
        {"TensorOpMultiplicand64bCrosswise",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand64bCrosswise>},
        {"TensorOpMultiplicandCongruous<128,4>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCongruous<128, 4>>},
        {"TensorOpMultiplicandCrosswise<128,4>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCrosswise<128, 4>>},
        {"VoltaTensorOpMultiplicandCongruous<16>",
         new VisualizeLayout<
             cutlass::layout::VoltaTensorOpMultiplicandCongruous<16>>},
        {"VoltaTensorOpMultiplicandCrosswise<16,32>",
         new VisualizeLayout<
             cutlass::layout::VoltaTensorOpMultiplicandCrosswise<16, 32>>}

    }
    /*
    std::map 是 C++ 标准库中的一个关联容器，用于存储键值对（key-value pairs）。emplace 是 C++11 引入的一种方法，用于在容器中直接构造元素，以提高性能和简化代码。


emplace 的作用
emplace 的主要作用是直接在容器中构造元素，而不是先构造一个临时对象再将其插入到容器中。这可以避免不必要的拷贝或移动操作，从而提高性能。


使用 emplace 的优势
避免不必要的拷贝和移动：
通过在容器内直接构造元素，可以避免额外的拷贝或移动操作。
提高效率：
在某些情况下，特别是当构造对象的代价较高时，emplace 可以显著提高性能。
简化代码：
emplace 允许直接传递构造函数的参数，而不需要先创建一个对象。

emplace 的用法
emplace 的用法类似于 insert，但它接受构造函数参数列表，并在容器中直接构造元素。


#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<int, std::string> myMap;

    // 使用 emplace 插入键值对
    myMap.emplace(1, "One");
    myMap.emplace(2, "Two");

    // 尝试插入一个已存在的键
    auto result = myMap.emplace(1, "Uno");
    if (!result.second) {
        std::cout << "Key 1 already exists with value: " << result.first->second << '\n';
    }

    // 打印 map 中的所有元素
    for (const auto& pair : myMap) {
        std::cout << pair.first << ": " << pair.second << '\n';
    }

    return 0;
}

代码说明
myMap.emplace(1, "One"): 直接在 myMap 中构造一个键值对 (1, "One")。
myMap.emplace(2, "Two"): 同样地，直接构造 (2, "Two")。
检查插入结果: emplace 返回一个 std::pair，其中 second 是一个布尔值，表示插入是否成功。如果 second 为 false，表示插入失败，可能是因为键已经存在。

注意事项
emplace 的行为类似于 insert，即如果键已经存在，它不会更新对应的值。
如果需要在键存在时更新值，可以使用 operator[] 或 insert_or_assign（C++17 引入）。

emplace 是一个强大而高效的工具，尤其在需要直接构造复杂对象时，非常有用。通过减少不必要的对象拷贝和移动，它可以帮助优化程序性能。
    */
    for (auto layout : layout_pairs) {
        layouts.emplace(std::string(layout.name), std::unique_ptr<VisualizeLayoutBase>(layout.ptr));
      }
}