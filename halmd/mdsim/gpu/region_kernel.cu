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

#include <halmd/mdsim/gpu/region_kernel.hpp>
#include <halmd/mdsim/gpu/box_kernel.cuh>
#include <halmd/utility/gpu/thread.cuh>

#include <halmd/mdsim/geometries/cuboid.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace region_kernel {

template <typename vector_type, typename geometry_type>
__global__ void compute_mask(
    float4 const* g_r
  , unsigned int nparticle
  , unsigned int* g_mask
  , geometry_type const geometry
  , vector_type box_length
)
{
    enum { dimension = vector_type::static_size };
    unsigned int const i = GTID;
    if(i >= nparticle)
        return;

    vector_type r;
    unsigned int type;
    tie(r, type) <<= g_r[i];

    // enforce periodic boundary conditions
    box_kernel::reduce_periodic(r, box_length);
    // 1 means the particle in in the selector, 0 means outside
    g_mask[i] = geometry(r) ? 1 : 0;
}

/**
 * generate ascending index sequence
 */
__global__ void gen_index(unsigned int* g_index, unsigned int nparticle)
{
    g_index[GTID] = (GTID < nparticle) ? GTID : 0;
}

__global__ void compute_bin_border(
    unsigned int* g_offset
  , unsigned int* const g_mask
  , unsigned int nparticle
)
{
    unsigned int const i = GTID;
    if(i >= nparticle-1)
        return;
    if (g_mask[i+1] > g_mask[i]) {
        *g_offset = i+1;
    }
}

} // namespace region_kernel

template<int dimension, typename geometry_type>
region_wrapper<dimension, geometry_type> const
region_wrapper<dimension, geometry_type>::kernel = {
    region_kernel::compute_mask
  , region_kernel::gen_index
  , region_kernel::compute_bin_border
};

template class region_wrapper<3, halmd::mdsim::geometries::cuboid<3, float> >;
template class region_wrapper<2, halmd::mdsim::geometries::cuboid<2, float> >;

} // namespace gpu
} // namespace mdsim
} // namespace halmd
