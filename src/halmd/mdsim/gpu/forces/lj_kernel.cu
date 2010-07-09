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

#include <halmd/mdsim/gpu/box_kernel.cuh>
#include <halmd/mdsim/gpu/force_kernel.cuh>
#include <halmd/mdsim/gpu/forces/lj_kernel.hpp>
#include <halmd/mdsim/gpu/particle_kernel.cuh>
#include <halmd/numeric/gpu/blas/dsfloat.cuh>
#include <halmd/numeric/gpu/blas/symmetric.cuh>
#include <halmd/numeric/gpu/blas/vector.cuh>
#include <halmd/utility/gpu/thread.cuh>

using namespace boost::mpl;
using namespace halmd::mdsim::gpu::particle_kernel;
using namespace halmd::numeric::gpu::blas;

namespace halmd
{
namespace mdsim { namespace gpu { namespace forces
{

template <size_t N>
struct dim_
{
    /** positions, types */
    static texture<float4, 1, cudaReadModeElementType> r;
    /** cubic box edgle length */
    static __constant__ typename if_c<N == 3, float3, float2>::type box_length;
};

// explicit instantiation
template class dim_<3>;
template class dim_<2>;

/** number of placeholders per neighbour list */
__constant__ unsigned int neighbour_size_;
/** neighbour list stride */
__constant__ unsigned int neighbour_stride_;
/** Lennard-Jones potential parameters */
texture<float4, 1, cudaReadModeElementType> ljparam_;

/**
 * Compute Lennard-Jones forces
 */
template <typename vector_type, typename gpu_vector_type>
__global__ void compute(
  gpu_vector_type* g_f,
  unsigned int* g_neighbour,
  float* g_en_pot,
  gpu_vector_type* g_virial)
{
    enum { D = vector_type::static_size };
    typedef typename vector_type::value_type value_type;
    unsigned int i = GTID;

    // load particle associated with this thread
    unsigned int type1;
    vector_type r1;
    tie(r1, type1) = untagged<vector_type>(tex1Dfetch(dim_<D>::r, i));

    // potential energy contribution
    float en_pot_ = 0;
    // virial contribution
    vector<float, (D - 1) * D / 2 + 1> virial_ = 0;
#ifdef USE_FORCE_DSFUN
    // force sum
    vector<dsfloat, D> f = 0;
#else
    vector_type f = 0;
#endif

    for (unsigned int k = 0; k < neighbour_size_; ++k) {
        // coalesced read from neighbour list
        unsigned int j = g_neighbour[k * neighbour_stride_ + i];
        // skip placeholder particles
        if (j == particle_kernel::PLACEHOLDER) {
            break;
        }

        // load particle
        unsigned int type2;
        vector_type r2;
        tie(r2, type2) = untagged<vector_type>(tex1Dfetch(dim_<D>::r, j));
        // Lennard-Jones potential parameters
        vector<float, 4> lj = tex1Dfetch(ljparam_, symmetric_matrix::lower_index(type1, type2));

        // particle distance vector
        vector_type r = r1 - r2;
        // enforce periodic boundary conditions
        box_kernel::reduce_periodic(r, static_cast<vector_type>(dim_<D>::box_length));
        // squared particle distance
        value_type rr = inner_prod(r, r);
        // enforce cutoff length
        if (rr >= lj[lj_kernel<D>::RR_CUT]) {
            return;
        }

        // compute Lennard-Jones force in reduced units
        value_type rri = lj[lj_kernel<D>::SIGMA2] / rr;
        value_type ri6 = rri * rri * rri;
        value_type fval = 48 * lj[lj_kernel<D>::EPSILON] * rri * ri6 * (ri6 - 0.5f) / lj[lj_kernel<D>::SIGMA2];
        value_type en_pot = 4 * lj[lj_kernel<D>::EPSILON] * ri6 * (ri6 - 1) - lj[lj_kernel<D>::EN_CUT];

        // virial equation sum
        virial_ += 0.5f * fval * force_kernel::virial_tensor(rr, r);
        // potential energy contribution of this particle
        en_pot_ += 0.5f * en_pot;
        // force from other particle acting on this particle
        f += fval * r;
    }

    g_f[i] = static_cast<vector_type>(f);
    g_en_pot[i] = en_pot_;
    g_virial[i] = virial_;
}

template <int D> typeof(lj_kernel<D>::r)
  lj_kernel<D>::r(forces::dim_<D>::r);
template <int D> typeof(lj_kernel<D>::box_length)
  lj_kernel<D>::box_length(forces::dim_<D>::box_length);
template <int D> typeof(lj_kernel<D>::neighbour_size)
  lj_kernel<D>::neighbour_size(forces::neighbour_size_);
template <int D> typeof(lj_kernel<D>::neighbour_stride)
  lj_kernel<D>::neighbour_stride(forces::neighbour_stride_);
template <int D> typeof(lj_kernel<D>::ljparam)
  lj_kernel<D>::ljparam(forces::ljparam_);
template <int D> typeof(lj_kernel<D>::compute)
  lj_kernel<D>::compute(forces::compute<vector<float, D> >);

}}} // namespace mdsim::gpu::forces

template class mdsim::gpu::forces::lj_kernel<3>;
template class mdsim::gpu::forces::lj_kernel<2>;

} // namespace halmd
