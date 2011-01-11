/*
 * Copyright © 2008-2010  Peter Colberg and Felix Höfling
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

#ifndef HALMD_MDSIM_GPU_FORCES_LENNARD_JONES_KERNEL_CUH
#define HALMD_MDSIM_GPU_FORCES_LENNARD_JONES_KERNEL_CUH

#include <halmd/mdsim/gpu/forces/lennard_jones_kernel.hpp>
#include <halmd/mdsim/gpu/forces/pair_trunc_kernel.cuh>
#include <halmd/numeric/blas/blas.hpp>
#include <halmd/utility/gpu/variant.cuh>

namespace halmd
{
namespace mdsim { namespace gpu { namespace forces
{
namespace lennard_jones_kernel
{

/** array of Lennard-Jones potential parameters for all combinations of particle types */
static __constant__ float4 param_;

/**
 * Lennard-Jones interaction of a pair of particles.
 */
class lennard_jones
{
public:
    /**
     * Construct Lennard-Jones pair interaction potential.
     *
     * Fetch potential parameters from texture cache for particle pair.
     *
     * @param type1 type of first interacting particle
     * @param type2 type of second interacting particle
     */
    HALMD_GPU_ENABLED lennard_jones(unsigned int type1, unsigned int type2)
      : pair_(
              param_
        ) {}

    /**
     * Check whether particles are in interaction range.
     *
     * @param rr squared distance between particles
     */
    template <typename float_type>
    HALMD_GPU_ENABLED bool within_range(float_type rr) const
    {
        return (rr < pair_[RR_CUT]);
    }

    /**
     * Compute force and potential for interaction.
     *
     * @param rr squared distance between particles
     * @returns tuple of absolute unit force and potential
     */
    template <typename float_type>
    HALMD_GPU_ENABLED tuple<float_type, float_type> operator()(float_type rr) const
    {
        float_type rri = pair_[SIGMA2] / rr;
        float_type ri6 = rri * rri * rri;
        float_type fval = 48 * pair_[EPSILON] * rri * ri6 * (ri6 - 0.5f) / pair_[SIGMA2];
        float_type en_pot = 4 * pair_[EPSILON] * ri6 * (ri6 - 1) - pair_[EN_CUT];

        return make_tuple(fval, en_pot);
    }

    /**
     * Compute hypervirial for interaction.
     *
     * @param rr squared distance between particles
     * @returns hypervirial contribution for this particle pair
     */
    template <typename float_type>
    HALMD_GPU_ENABLED float_type hypervirial(float_type rr) const
    {
        float_type rri = pair_[SIGMA2] / rr;
        float_type ri6 = rri * rri * rri;
        return 576 * pair_[EPSILON] * ri6 * (ri6 - 0.25f);
    }

private:
    /** potential parameters for particle pair */
    fixed_vector<float, 4> pair_;
};

} // namespace lennard_jones_kernel

cuda::symbol<float4> lennard_jones_wrapper::param = lennard_jones_kernel::param_;

template class pair_trunc_wrapper<3, lennard_jones_kernel::lennard_jones>;
template class pair_trunc_wrapper<2, lennard_jones_kernel::lennard_jones>;

}}} // namespace mdsim::gpu::forces

} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_FORCES_LENNARD_JONES_KERNEL_CUH */
