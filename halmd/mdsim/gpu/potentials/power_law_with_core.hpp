/*
 * Copyright © 2011  Michael Kopp and Felix Höfling
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

#ifndef HALMD_MDSIM_GPU_POTENTIALS_POWER_LAW_WITH_CORE_HPP
#define HALMD_MDSIM_GPU_POTENTIALS_POWER_LAW_WITH_CORE_HPP

#include <boost/make_shared.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <cuda_wrapper/cuda_wrapper.hpp>
#include <lua.hpp>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/potentials/power_law_with_core_kernel.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {

/**
 * define potential of power law with core and its parameters
 */
template <typename float_type>
class power_law_with_core
{
public:
    typedef power_law_with_core_kernel::power_law_with_core gpu_potential_type;
    typedef boost::numeric::ublas::symmetric_matrix<float_type, boost::numeric::ublas::lower> matrix_type;
    typedef boost::numeric::ublas::symmetric_matrix<unsigned, boost::numeric::ublas::lower> uint_matrix_type;
    typedef logger logger_type;

    static char const* module_name() { return "power_law_with_core"; }

    static void luaopen(lua_State* L);

    power_law_with_core(
        unsigned ntype
      , boost::array<float, 3> const& cutoff
      , boost::array<float, 3> const& core
      , boost::array<float, 3> const& epsilon
      , boost::array<float, 3> const& sigma
      , boost::array<unsigned, 3> const& index
      , boost::shared_ptr<logger_type> logger = boost::make_shared<logger_type>()
    );

    /** bind textures before kernel invocation */
    void bind_textures() const
    {
        power_law_with_core_wrapper::param.bind(g_param_);
        power_law_with_core_wrapper::rr_en_cut.bind(g_rr_en_cut_);
    }

    matrix_type const& r_cut() const
    {
        return r_cut_;
    }

    float_type r_cut(unsigned a, unsigned b) const
    {
        return r_cut_(a, b);
    }

    float_type rr_cut(unsigned a, unsigned b) const
    {
        return rr_cut_(a, b);
    }

    matrix_type const& r_cut_sigma() const
    {
        return r_cut_sigma_;
    }

    matrix_type const& r_core_sigma() const
    {
        return r_core_sigma_;
    }

    matrix_type const& epsilon() const
    {
        return epsilon_;
    }

    matrix_type const& sigma() const
    {
        return sigma_;
    }

    uint_matrix_type const& index() const
    {
        return index_;
    }

private:
    /** potential well depths in MD units */
    matrix_type epsilon_;
    /** pair separation in MD units */
    matrix_type sigma_;
    /** power law index */
    uint_matrix_type index_;
    /** cutoff length in units of sigma */
    matrix_type r_cut_sigma_;
    /** cutoff length in MD units */
    matrix_type r_cut_;
    /** square of cutoff length */
    matrix_type rr_cut_;
    /** core radius in units of sigma */
    matrix_type r_core_sigma_;
    /** square of pair separation */
    matrix_type sigma2_;
    /** potential energy at cutoff length in MD units */
    matrix_type en_cut_;
    /** potential parameters at CUDA device */
    cuda::vector<float4> g_param_;
    /** squared cutoff radius and energy shift at CUDA device */
    cuda::vector<float2> g_rr_en_cut_;
    /** module logger */
    boost::shared_ptr<logger_type> logger_;
};

} // namespace potentials
} // namespace gpu
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_POTENTIALS_POWER_LAW_WITH_CORE_HPP */
