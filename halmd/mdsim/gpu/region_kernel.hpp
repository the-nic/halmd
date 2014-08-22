/*
 * Copyright © 2014 Nicolas Höft
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

#ifndef HALMD_MDSIM_GPU_REGION_KERNEL_HPP
#define HALMD_MDSIM_GPU_REGION_KERNEL_HPP

#include <halmd/numeric/blas/fixed_vector.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

template<int dimension, typename geometry_type>
struct region_wrapper
{
    typedef fixed_vector<float, dimension> vector_type;

    /** create a mask for particles within/outside the region */
    cuda::function<void (
        float4 const* // position
      , unsigned int  // nparticle
      , unsigned int* // mask
      , geometry_type const
      , vector_type  // box length
    )> compute_mask;

    /** generate ascending index sequence */
    cuda::function<void (
        unsigned int*  // index sequence
      , unsigned int   // nparticle
    )> gen_index;

    /** calculate position where the mask 0…01…1 switches from 0->1 */
    cuda::function<void (
        unsigned int*       // offset
      , unsigned int* const // mask
      , unsigned int        // nparticle
    )> compute_bin_border;

    static region_wrapper const kernel;
};

} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_REGION_KERNEL_HPP */
