/*
#pragma once 是一个预处理指令，用于防止头文件被多次包含。它是一种防止重复包含（include guard）的简便方法，广泛用于 C 和 C++ 编程中。


作用
当一个头文件被多次包含时，可能会导致重复定义错误或增加编译时间。#pragma once 指令告诉编译器在同一个编译单元内只包含该头文件一次，从而避免这些问题。


工作原理
唯一性: 当编译器遇到 #pragma once 时，它会记录下该头文件的路径。如果该文件再次被包含，编译器会跳过它。
效率: 相比于传统的 include guard（使用宏定义），#pragma once 可能会更高效，因为它不需要进行预处理器的条件编译判断。
*/

#pragma once

#include <vector>
#include <iostream>

// Cutlass command line parser
#include "cutlass/util/command_line.h"

class Options {
public:

  bool help;
  bool good;
  std::vector<int> extent;          ///< extent of tile to fill
  std::vector<int> stride;          ///< stride vector for layout function
  std::vector<int> output_shape;    ///< output shape
  int vectorize;                    ///< sequences of consecutive output elements are concatenated into a vector
                                    ///  if, and only if, they were consecutive in source memory

public:

  /// Options
  Options(): 
    help(false),
    good(true),
    extent({32, 8}),
    stride({32}),
    output_shape({16, 8}), 
    vectorize(1) { 

  }

  /// Constructs from command line parser
  Options(cutlass::CommandLine const & cmd_line): help(false), good(true) {

    if (cmd_line.check_cmd_line_flag("help") ||
        cmd_line.check_cmd_line_flag("h")) {

      help = true;
    }

    if (cmd_line.check_cmd_line_flag("extent")) {
      cmd_line.get_cmd_line_arguments("extent", extent);
    }
    else {
      extent = {32, 8};
    }

    if (cmd_line.check_cmd_line_flag("stride")) {
      cmd_line.get_cmd_line_arguments("stride", stride);
    }
    
    int default_output_shape[] = {16, 8}; 

    if (cmd_line.check_cmd_line_flag("output-shape")) {
      cmd_line.get_cmd_line_arguments("output-shape", output_shape);
    }

    for (int i = int(output_shape.size()); i < 2; ++i) {
      output_shape.push_back(default_output_shape[i]);
    }

    if (cmd_line.check_cmd_line_flag("vectorize")) {
      cmd_line.get_cmd_line_argument("vectorize", vectorize);
    }
    else {
      vectorize = 1;
    }

    if (output_shape.front() % vectorize) {

      std::cerr << "Error: --vectorize=" << vectorize 
        << " must divide contiguous elements in --output-shape="
        << output_shape.at(0) << "," << output_shape.at(1) << std::endl;

      good = false;
    }
  }

  /// Prints usage statement
  static void print_usage(std::ostream &out) {
    out
      << "  Options:\n"
      << "    --help                              Displays this help message.\n"
      << "    --extent=<extent>                   Specifies the layout-specific extent (as comma-delimited array).\n"
      << "    --stride=<stride>                   Specifies the layout-specific stride vector (comma-delimited array)\n"
      << "    --output-shape=<extent>             Specifies the dimensions of a row-major output matrix. \n"
      << "    --vectorize=<vector length>         If possible, vectorizes the output into vectors of consecutive elements\n";
  }
};


