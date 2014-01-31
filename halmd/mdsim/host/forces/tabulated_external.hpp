/*
 * Copyright © 2013      Nicolas Höft
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

#ifndef HALMD_MDSIM_HOST_FORCES_TABLULATED_EXTERNAL_HPP
#define HALMD_MDSIM_HOST_FORCES_TABLULATED_EXTERNAL_HPP

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/utility/profiler.hpp>

#include <lua.hpp>

#include <memory>
#include <tuple>

namespace halmd {
namespace mdsim {
namespace host {
namespace forces {

/**
 * class template for modules implementing an external force based on pretabulated values
 */
template <int dimension, typename float_type, typename interpolation_type>
class tabulated_external
{
public:
    typedef float_type coefficient_value_type;
    typedef raw_array<coefficient_value_type> coefficient_array_type;

    typedef particle<dimension, float_type> particle_type;
    typedef box<dimension> box_type;
    typedef typename particle_type::position_type position_type;
    typedef logger logger_type;

    tabulated_external(
        std::shared_ptr<particle_type> particle
      , std::shared_ptr<box_type const> box
      , std::shared_ptr<interpolation_type> interpolation
      , std::shared_ptr<logger_type> logger = std::make_shared<logger_type>()
    );

    /**
     * Test if the cache is up-to-date and if not, inform the particle
     * module about it (usually done at on_prepend_force())
     */
    void check_cache();

    /**
     * Apply the force to the particles
     */
    void apply();

    /**
     * Return the const reference of the interpolation coefficients
     */
    coefficient_array_type const& coefficients() const
    {
        return coefficients_;
    }

    /**
     * Return the reference of the interpolation coefficients
     */
    coefficient_array_type& coefficients()
    {
        return coefficients_;
    }

    /**
     * Return total number of needed coeffcients for interpolation
     */
    size_t ncoefficients() const
    {
        return coefficients_.size();
    }

    /**
     * Bind class to Lua.
     */
    static void luaopen(lua_State* L);

private:
    typedef typename particle_type::position_array_type position_array_type;
    typedef typename particle_type::force_type force_type;
    typedef fixed_vector<unsigned int, dimension> index_type;
    typedef typename particle_type::size_type size_type;

    /** compute forces */
    void compute_();
    /** compute forces with auxiliary variables */
    void compute_aux_();

    /** state of system */
    std::shared_ptr<particle_type> particle_;
    /** simulation domain */
    std::shared_ptr<box_type const> box_;
    /** edges of the precalculated grid */
    std::shared_ptr<position_type const> grid_edges_;
    /** interpolation functor */
    std::shared_ptr<interpolation_type const> interpolation_;
    /** module logger */
    std::shared_ptr<logger_type> logger_;

    /** coeffcients for the interpolation scheme */
    coefficient_array_type coefficients_;

    /** cache observer of net force per particle */
    cache<> force_cache_;
    /** cache observer of auxiliary variables */
    cache<> aux_cache_;

    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;
    typedef typename profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type compute;
    };

    /** profiling runtime accumulators */
    runtime runtime_;
};

} // namespace forces
} // namespace host
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_FORCES_TABLULATED_EXTERNAL_HPP */