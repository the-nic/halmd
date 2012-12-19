/*
 * Copyright © 2008-2011  Peter Colberg and Felix Höfling
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

#ifndef HALMD_MDSIM_GPU_INTEGRATORS_VERLET_HPP
#define HALMD_MDSIM_GPU_INTEGRATORS_VERLET_HPP

#include <lua.hpp>
#include <memory>

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <halmd/io/logger.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/gpu/force.hpp>
#include <halmd/mdsim/gpu/integrators/verlet_kernel.hpp>
#include <halmd/mdsim/gpu/particle.hpp>
#include <halmd/mdsim/gpu/particle_group.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace integrators {

template <int dimension, typename float_type>
class verlet
{
public:
    typedef particle<dimension, float_type> particle_type;
    typedef particle_group particle_group_type;
    typedef force<dimension, float_type> force_type;
    typedef box<dimension> box_type;
    typedef logger logger_type;
    typedef typename particle_type::vector_type vector_type;

    static void luaopen(lua_State* L);

    verlet(
        std::shared_ptr<particle_type> particle
      , std::shared_ptr<particle_group_type> group
      , std::shared_ptr<force_type> force
      , std::shared_ptr<box_type const> box
      , double timestep
      , std::shared_ptr<logger_type> logger = std::make_shared<logger_type>()
    );

    /**
     * Copy net forces to buffer.
     */
    void acquire_net_force();

    void integrate();
    void finalize();
    void set_timestep(double timestep);

    //! returns integration time-step
    double timestep() const
    {
        return timestep_;
    }

private:
    typedef typename particle_type::position_array_type position_array_type;
    typedef typename particle_type::image_array_type image_array_type;
    typedef typename particle_type::velocity_array_type velocity_array_type;
    typedef typename force_type::net_force_array_type net_force_array_type;
    typedef typename particle_group_type::array_type group_array_type;

    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;
    typedef typename profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type integrate;
        accumulator_type finalize;
    };

    /** system state */
    std::shared_ptr<particle_type> particle_;
    /** particle group */
    std::shared_ptr<particle_group_type> group_;
    /** particle forces */
    std::shared_ptr<force_type> force_;
    /** simulation domain */
    std::shared_ptr<box_type const> box_;
    /** module logger */
    std::shared_ptr<logger_type> logger_;
    /** CUDA C++ wrapper */
    verlet_wrapper<dimension> const* wrapper_;
    /** integration time-step */
    float_type timestep_;
    /** buffer of net forces */
    net_force_array_type net_force_;
    /** cache observer of net forces */
    cache<> net_force_cache_;
    /** profiling runtime accumulators */
    runtime runtime_;
};

} // namespace mdsim
} // namespace gpu
} // namespace integrators
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_INTEGRATORS_VERLET_HPP */
