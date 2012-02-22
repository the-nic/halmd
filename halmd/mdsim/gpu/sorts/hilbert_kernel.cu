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

#include <boost/mpl/if.hpp>
#include <float.h>

#include <halmd/algorithm/gpu/bits.cuh>
#include <halmd/mdsim/gpu/particle_kernel.cuh>
#include <halmd/mdsim/gpu/sorts/hilbert_kernel.hpp>
#include <halmd/mdsim/sorts/hilbert_kernel.hpp>
#include <halmd/numeric/blas/blas.hpp>
#include <halmd/utility/gpu/thread.cuh>
#include <halmd/utility/gpu/variant.cuh>

using namespace halmd::algorithm::gpu;
using namespace halmd::mdsim::gpu::particle_kernel;
using namespace halmd::utility::gpu;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace sorts {
namespace hilbert_kernel {

/** Hilbert space-filling curve recursion depth */
__constant__ unsigned int depth_;
/** cubic box edgle length */
__constant__ variant<map<pair<int_<3>, float3>, pair<int_<2>, float2> > > box_length_;

/**
 * generate Hilbert space-filling curve
 */
template <typename vector_type>
__global__ void map(float4 const* g_r, unsigned int* g_sfc)
{
    enum { dimension = vector_type::static_size };

    //
    // We need to avoid ambiguities during the assignment of a particle
    // to a subcell, i.e. the particle position should never lie on an
    // edge or corner of multiple subcells, or the algorithm will have
    // trouble converging to a definite Hilbert curve.
    //
    // Therefore, we use a simple cubic lattice of predefined dimensions
    // according to the number of cells at the deepest recursion level,
    // and round the particle position to the nearest center of a cell.
    //

    unsigned int type;
    vector_type r;
    tie(r, type) = untagged<vector_type>(g_r[GTID]);
    vector_type L = get<dimension>(box_length_);
    r = element_div(r, L);

    // compute Hilbert code for particle
    g_sfc[GTID] = mdsim::sorts::hilbert_kernel::map(r, depth_);
}

/**
 * generate ascending index sequence
 */
__global__ void gen_index(unsigned int* g_index)
{
    g_index[GTID] = GTID;
}

} // namespace hilbert_kernel

template <int dimension>
hilbert_wrapper<dimension> const hilbert_wrapper<dimension>::kernel = {
    hilbert_kernel::depth_
  , get<dimension>(hilbert_kernel::box_length_)
  , hilbert_kernel::map<fixed_vector<float, dimension> >
  , hilbert_kernel::gen_index
};

// explicit instantiation
template class hilbert_wrapper<3>;
template class hilbert_wrapper<2>;

} // namespace mdsim
} // namespace gpu
} // namespace sorts
} // namespace halmd