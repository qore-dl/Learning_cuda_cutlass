/*! \file
  \brief CUTLASS layout visualization example
*/

#pragma once

#include <map>
#include <memory>

#include "options.h"

/*
在 C++ 中，virtual 关键字用于声明虚函数，主要用于实现多态性（polymorphism）。多态性是面向对象编程中的一个核心概念，它允许程序在运行时决定调用哪个函数实现。这对于实现灵活和可扩展的代码结构非常重要。


作用
实现多态性:
当基类中某个函数被声明为虚函数时，派生类可以重写（override）该函数。
在使用基类指针或引用指向派生类对象时，调用虚函数会根据对象的实际类型（而不是指针或引用的类型）来决定调用哪个函数实现，这就是多态性。
支持动态绑定:
动态绑定（或晚绑定）是指在运行时决定调用哪个函数版本。使用虚函数表（vtable）机制实现。
非虚函数则是静态绑定（或早绑定），在编译时决定调用哪个函数。

关键点
虚函数表（vtable）:
每个包含虚函数的类都有一个虚函数表，表中存储着指向该类各个虚函数实现的指针。
对象中有一个虚指针（vptr），指向所属类的虚函数表。
虚析构函数:
当一个类有虚函数时，通常需要定义虚析构函数。这确保在删除基类指针时，派生类的析构函数也会被正确调用，防止资源泄漏。
override 关键字:
C++11 引入的 override 关键字用于显式说明派生类中的某个函数是重写基类的虚函数。这有助于编译器检查是否正确地重写了基类函数。
*/

/////////////////////////////////////////////////////////////////////////////////////////////////
struct VisualizeLayoutBase{
    virtual bool visualize(Options const &) = 0;
    virtual bool verify(bool verbose,std::ostream &out) = 0;
    virtual bool print_csv(std::ostream &out, char delim = '|', char new_line = '\n') = 0;
    virtual std::ostream &print_help(std::ostream &out){
        return out;
    }
    virtual ~VisualizeLayoutBase() { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

void RegisterLayouts(std::map<std::string, std::unique_ptr<VisualizeLayoutBase> > &layouts);

/////////////////////////////////////////////////////////////////////////////////////////////////
