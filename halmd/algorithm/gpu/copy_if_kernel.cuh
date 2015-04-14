/*
 * Copyright © 2015 Nicolas Höft
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HALMD_ALGORITHM_GPU_COPY_IF_KERNEL_CUH
#define HALMD_ALGORITHM_GPU_COPY_IF_KERNEL_CUH

#include <cub/device/device_select.cuh>

#include <cuda_wrapper/error.hpp>
#include <cuda_wrapper/vector.hpp>

#include <halmd/algorithm/gpu/copy_if_kernel.hpp>

namespace halmd {
namespace algorithm {
namespace gpu {
namespace copy_if_kernel {

template <typename T, typename Predicate>
unsigned int copy_if(T* g_input, unsigned int size, Predicate predicate, T* g_output)
{
    cuda::vector<char> g_temp;
    // allocate gpu memory to store total number of selected elements
    cuda::vector<int> g_output_len(1);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    CUDA_CALL(cub::DeviceSelect::If(
        0
      , temp_storage_bytes
      , g_input
      , g_output
      , &*g_output_len.begin()
      , size
      , predicate
    ));

    g_temp.resize(temp_storage_bytes);

    // copy the elements according to the predicate
    CUDA_CALL(cub::DeviceSelect::If(
        &*g_temp.begin()
      , temp_storage_bytes
      , g_input
      , g_output
      , &*g_output_len.begin()
      , size
      , predicate
    ));

    // the output length resides in gpu, copy it to host memory
    int h_output_len;
    CUDA_CALL(cudaMemcpy(
        &h_output_len
      , &*g_output_len.begin()
      , sizeof(int)
      , cudaMemcpyDeviceToHost
    ));
    //cuda::copy(g_output_len.begin(), g_output_len.begin()+1, &h_output_len);
    return h_output_len;
}

} // namespace copy_if_kernel

/**
 * CUDA C++ wrapper
 */
template <typename T, typename Predicate>
copy_if_wrapper<T, Predicate> const copy_if_wrapper<T, Predicate>::kernel = {
   copy_if_kernel::copy_if<T, Predicate>
};

} // namespace gpu
} // namespace algorithm
} // namespace halmd

#endif /* ! HALMD_ALGORITHM_GPU_COPY_IF_KERNEL_CUH */
