# Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


if (CUTLASS_NVCC_ARCHS MATCHES 100a)
add_custom_target(
  cutlass_test_unit_gemm_device_sm100_tensorop_narrow_precision
  DEPENDS
  cutlass_test_unit_gemm_device_tensorop_sm100_f6f4xf6f4
  cutlass_test_unit_gemm_device_tensorop_sm100_f6f4xf8
  cutlass_test_unit_gemm_device_tensorop_sm100_f8xf6f4
)

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm100_f6f4xf6f4

  BATCH_SOURCES ON
  BATCH_SIZE 1

  f6f4_f6f4_void_f32_tn_layout.cu
  f6f4_f6f4_void_f32_nn_layout.cu
  f6f4_f6f4_void_f32_nt_layout.cu
  f6f4_f6f4_void_f32_tt_layout.cu
  )

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm100_f6f4xf8

  BATCH_SOURCES ON
  BATCH_SIZE 1

  f6f4_f8_void_f32_tn_layout.cu
  f6f4_f8_void_f32_nt_layout.cu
  )

cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm100_f8xf6f4

  BATCH_SOURCES ON
  BATCH_SIZE 1

  f8_f6f4_void_f32_tn_layout.cu
  f8_f6f4_void_f32_nt_layout.cu
  )
endif()
