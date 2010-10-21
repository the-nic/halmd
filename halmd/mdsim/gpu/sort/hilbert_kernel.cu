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
#include <halmd/mdsim/gpu/sort/hilbert_kernel.hpp>
#include <halmd/numeric/blas/blas.hpp>
#include <halmd/utility/gpu/thread.cuh>
#include <halmd/utility/gpu/variant.cuh>

using namespace halmd::algorithm::gpu;
using namespace halmd::mdsim::gpu::particle_kernel;
using namespace halmd::utility::gpu;

namespace halmd
{
namespace mdsim { namespace gpu
{
namespace hilbert_kernel
{

/** Hilbert space-filling curve recursion depth */
__constant__ unsigned int depth_;
/** cubic box edgle length */
__constant__ variant<map<pair<int_<3>, float3>, pair<int_<2>, float2> > > box_length_;

/**
 * swap Hilbert spacing-filling curve vertices
 */
__device__ void swap_vertex(uint& v, uint& a, uint& b, uint const& mask)
{
    // swap bits comprising Hilbert codes in vertex-to-code lookup table
    uint const va = ((v >> a) & mask);
    uint const vb = ((v >> b) & mask);
    v = v ^ (va << a) ^ (vb << b) ^ (va << b) ^ (vb << a);
    // update code-to-vertex lookup table
    swap(a, b);
}

/**
 * map 3-dimensional point to 1-dimensional point on Hilbert space curve
 */
__device__ unsigned int _map(fixed_vector<float, 3> r)
{
    //
    // Jun Wang & Jie Shan, Space-Filling Curve Based Point Clouds Index,
    // GeoComputation, 2005
    //

    // Hilbert code for particle
    unsigned int hcode = 0;
    // Hilbert code-to-vertex lookup table
    uint a = 21;
    uint b = 18;
    uint c = 12;
    uint d = 15;
    uint e = 3;
    uint f = 0;
    uint g = 6;
    uint h = 9;
    // Hilbert vertex-to-code lookup table
    uint vc = 1U << b ^ 2U << c ^ 3U << d ^ 4U << e ^ 5U << f ^ 6U << g ^ 7U << h;

#define MASK ((1 << 3) - 1)

    // 32-bit integer for 3D Hilbert code allows a maximum of 10 levels
    for (unsigned int i = 0; i < depth_; ++i) {
        // determine Hilbert vertex closest to particle
        fixed_vector<unsigned int, 3> x;
        x[0] = __signbitf(r[0]) & 1;
        x[1] = __signbitf(r[1]) & 1;
        x[2] = __signbitf(r[2]) & 1;
        // lookup Hilbert code
        const uint v = (vc >> (3 * (x[0] + (x[1] << 1) + (x[2] << 2))) & MASK);

        // scale particle coordinates to subcell
        r = 2 * r - (fixed_vector<float, 3>(0.5f) - fixed_vector<float, 3>(x));
        // apply permutation rule according to Hilbert code
        if (v == 0) {
            swap_vertex(vc, b, h, MASK);
            swap_vertex(vc, c, e, MASK);
        }
        else if (v == 1 || v == 2) {
            swap_vertex(vc, c, g, MASK);
            swap_vertex(vc, d, h, MASK);
        }
        else if (v == 3 || v == 4) {
            swap_vertex(vc, a, c, MASK);
#ifdef USE_HILBERT_ALT_3D
            swap_vertex(vc, b, d, MASK);
            swap_vertex(vc, e, g, MASK);
#endif
            swap_vertex(vc, f, h, MASK);
        }
        else if (v == 5 || v == 6) {
            swap_vertex(vc, a, e, MASK);
            swap_vertex(vc, b, f, MASK);
        }
        else if (v == 7) {
            swap_vertex(vc, a, g, MASK);
            swap_vertex(vc, d, f, MASK);
        }

        // add vertex code to partial Hilbert code
        hcode = (hcode << 3) + v;
    }
#undef MASK
    return hcode;
}

/**
 * map 2-dimensional point to 1-dimensional point on Hilbert space curve
 */
__device__ unsigned int _map(fixed_vector<float, 2> r)
{
    // Hilbert code for particle
    unsigned int hcode = 0;
    // Hilbert code-to-vertex lookup table
    uint a = 6;
    uint b = 4;
    uint c = 0;
    uint d = 2;
    // Hilbert vertex-to-code lookup table
    uint vc = 1U << b ^ 2U << c ^ 3U << d;

#define MASK ((1 << 2) - 1)

    // 32-bit integer for 2D Hilbert code allows a maximum of 16 levels
    for (unsigned int i = 0; i < depth_; ++i) {
        // determine Hilbert vertex closest to particle
        fixed_vector<unsigned int, 2> x;
        x[0] = __signbitf(r[0]) & 1;
        x[1] = __signbitf(r[1]) & 1;
        // lookup Hilbert code
        const uint v = (vc >> (2 * (x[0] + (x[1] << 1))) & MASK);

        // scale particle coordinates to subcell
        r = 2 * r - (fixed_vector<float, 2>(0.5f) - fixed_vector<float, 2>(x));
        // apply permutation rule according to Hilbert code
        if (v == 0) {
            swap_vertex(vc, b, d, MASK);
        }
        else if (v == 3) {
            swap_vertex(vc, a, c, MASK);
        }

        // add vertex code to partial Hilbert code
        hcode = (hcode << 2) + v;
    }
#undef MASK
    return hcode;
}

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
    // Hilbert cells per dimension at deepest recursion level
    uint const n = 1UL << depth_;
    // fractional index of particle's Hilbert cell in [0, n)
    r = n * (saturate(element_div(r, L)) * (1.f - FLT_EPSILON));

    // round particle position to center of cell in unit coordinates
    r = (floor(r) + vector_type(0.5f)) / n;
    // use symmetric coordinates
    r -= vector_type(0.5f);

    // compute Hilbert code for particle
    g_sfc[GTID] = _map(r);
}

} // namespace hilbert_kernel

template <int dimension>
hilbert_wrapper<dimension> const hilbert_wrapper<dimension>::kernel = {
    hilbert_kernel::depth_
  , get<dimension>(hilbert_kernel::box_length_)
  , hilbert_kernel::map<fixed_vector<float, dimension> >
};

// explicit instantiation
template class hilbert_wrapper<3>;
template class hilbert_wrapper<2>;

}} // namespace mdsim::gpu

} // namespace halmd