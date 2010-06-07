/*
 * Copyright © 2008-2010  Peter Colberg
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

#ifndef HALMD_MDSIM_GPU_FORCES_LJ_WRAPPER_CUH
#define HALMD_MDSIM_GPU_FORCES_LJ_WRAPPER_CUH

#include <boost/mpl/if.hpp>

#include <cuda_wrapper.hpp>
#include <halmd/mdsim/gpu/forces/lj_kernel.cuh>

namespace halmd { namespace mdsim { namespace gpu { namespace forces
{

template <size_t N>
struct lj_wrapper
{
    typedef typename boost::mpl::if_c<N == 3, float4, float2>::type coalesced_vector_type;
    typedef typename boost::mpl::if_c<N == 3, float3, float2>::type vector_type;

    /** positions, types */
    static cuda::texture<float4> r;
    /** cubic box edgle length */
    static cuda::symbol<vector_type> box_length;
    /** number of placeholders per neighbour list */
    static cuda::symbol<unsigned int> neighbour_size;
    /** neighbour list stride */
    static cuda::symbol<unsigned int> neighbour_stride;
    /** Lennard-Jones potential parameters */
    static cuda::texture<float4> ljparam;
    /** compute Lennard-Jones forces */
    static cuda::function<void (coalesced_vector_type*, unsigned int*, float*, coalesced_vector_type*)> compute;
};

}}}} // namespace halmd::mdsim::gpu::forces

#endif /* ! HALMD_MDSIM_GPU_FORCES_LJ_WRAPPER_CUH */
