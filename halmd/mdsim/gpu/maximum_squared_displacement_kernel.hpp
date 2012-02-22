/*
 * Copyright © 2008-2011  Peter Colberg
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

#ifndef HALMD_MDSIM_GPU_MAXIMUM_SQUARED_DISPLACEMENT_KERNEL_HPP
#define HALMD_MDSIM_GPU_MAXIMUM_SQUARED_DISPLACEMENT_KERNEL_HPP

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <halmd/mdsim/type_traits.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

template <int dimension>
struct maximum_squared_displacement_wrapper
{
    typedef typename type_traits<dimension, float>::gpu::vector_type vector_type;
    typedef cuda::function<void (float4 const* g_r, float4 const* g_r0, float* g_rr)> displacement_impl_type;

    /** number of particles in simulation box */
    cuda::symbol<unsigned int> nbox;
    /** cubic box edgle length */
    cuda::symbol<vector_type> box_length;
    /** maximum squared particle displacement */
    displacement_impl_type displacement_impl[5];

    static maximum_squared_displacement_wrapper kernel;
};

template <int dimension>
maximum_squared_displacement_wrapper<dimension> const& get_maximum_squared_displacement_kernel()
{
    return maximum_squared_displacement_wrapper<dimension>::kernel;
}

} // namespace mdsim
} // namespace gpu
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_MAXIMUM_SQUARED_DISPLACEMENT_KERNEL_HPP */