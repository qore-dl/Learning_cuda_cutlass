# CUDA 基础学习与cutlass 样例学习的进度汇总

目前我的学习主要包括两部分：

1. 对 CUDA C 编程基础内容的学习，包括：基础使用、编程模型、全局内存、共享内存、常量内存、流与并发等内容。

2. 对 Cutlass 高性能计算库的学习，尤其关注 Ampere 和 Hopper 架构上的 example 的学习（包括tensorop、stream-k、混合精度乘法等）。

## CUDA C 编程基础学习

首先，我根据 Nvidia 的[CUDA 编程指南](./cuda_basic_learning/NVIDIA_CUDA_编程指南.pdf) 从编程模型、内存管理、并发机制等方面对 CUDA C 编程的基础知识进行了学习。详细的样例与概述、总结等可见 [cuda_basic_learning](./cuda_basic_learning/)

## Cutlass Example 学习

### 整体计划
本次学习使用的版本是Cutlass 3.9.2 版本，在初步阅读 [Cutlass官方文档](./cutlass-v3.9.2-learning/README.md) 后，我根据其提供的 Example 来进行学习，目前学习的目标是复现其 [Example 69](./cutlass-v3.9.2-learning/examples/69_hopper_mixed_dtype_grouped_gemm/README.md) 中展示的混合精度乘法。根据其文档的建议以及我对Cutlass的学习进度，需要实现对 [Example 0](./cutlass-v3.9.2-learning/learning/00_basic_gemm/) ~ [Example 15](./cutlass-v3.9.2-learning/learning/15_ampere_sparse_tensorop_gemm/) 以及 Example [47](./cutlass-v3.9.2-learning/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu), [48](./cutlass-v3.9.2-learning/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu), [55](./cutlass-v3.9.2-learning/examples/55_hopper_mixed_dtype_gemm/) 的学习，其中 Example [69](./cutlass-v3.9.2-learning/examples/69_hopper_mixed_dtype_grouped_gemm/README.md) 可以视为 Example [55](./cutlass-v3.9.2-learning/examples/55_hopper_mixed_dtype_gemm/) 的扩展版本。

### 当前进度
当前已完成了Example 0 ~ 15 的学习，复现并跑通了相关实例 (见 [cutlass-v3.9.2-learning/learning/](./cutlass-v3.9.2-learning/learning/)),在阅读了[Nvidia Stream-K](./cutlass-v3.9.2-learning/papers/Stream-K.pdf)的论文后，理解了如图1和图2中所示的Sliced-K (图中为Data-parallel), Split-K (图中为Fixed-split) 和 Stream-K的数据tile与线程块映射方式，目前正在复现 Example 47,本周会复现完成Example 47,48，并使用两周左右的时间复现 Example 55 与 69。在完成新的Example复现后会更新本仓库，并更新文档。

![SlicedK](./cutlass-v3.9.2-learning/images/SlicedK.png "图 1: 在假设具有4个SM的GPU上，采用Sliced-K(Data-parallel)策略计算问题规模为384 × 384 × 128的GEMM的执行计划")

![SlicedK](./cutlass-v3.9.2-learning/images/Split-K-VS-Stream-K.png "图2: 在假设具有4个SM的GPU上，采用Split-K(Fixed-split)和Stream-K策略计算问题规模为384 × 384 × 128的GEMM的执行计划")

### 未来计划
在完成 Example 69 的复现后，将重点将 Cutlass 在 Hopper 架构下的example 复现完成，并更深入了解 Hopper 架构。





