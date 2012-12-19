/*
 * Copyright © 2008-2012  Peter Colberg
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

#include <algorithm>
#include <cmath>
#include <memory>

#include <halmd/mdsim/host/integrators/verlet.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace host {
namespace integrators {

template <int dimension, typename float_type>
verlet<dimension, float_type>::verlet(
    std::shared_ptr<particle_type> particle
  , std::shared_ptr<particle_group_type> group
  , std::shared_ptr<force_type> force
  , std::shared_ptr<box_type const> box
  , double timestep
  , std::shared_ptr<logger_type> logger
)
  // dependency injection
  : particle_(particle)
  , group_(group)
  , force_(force)
  , box_(box)
  , logger_(logger)
  , net_force_(particle_->nparticle())
{
    set_timestep(timestep);
}

template <int dimension, typename float_type>
void verlet<dimension, float_type>::set_timestep(double timestep)
{
    timestep_ = timestep;
    timestep_half_ = 0.5 * timestep;
}

template <int dimension, typename float_type>
void verlet<dimension, float_type>::acquire_net_force()
{
    cache<net_force_array_type> const& net_force_cache = force_->net_force();

    if (net_force_cache_ != net_force_cache) {
        cache_proxy<net_force_array_type const> net_force = net_force_cache;

        LOG_TRACE("copy net forces to buffer");

        std::copy(net_force->begin(), net_force->end(), net_force_.begin());

        net_force_cache_ = net_force_cache;
    }
}

/**
 * First leapfrog half-step of velocity-Verlet algorithm
 */
template <int dimension, typename float_type>
void verlet<dimension, float_type>::integrate()
{
    cache_proxy<mass_array_type const> mass = particle_->mass();
    cache_proxy<position_array_type> position = particle_->position();
    cache_proxy<image_array_type> image = particle_->image();
    cache_proxy<velocity_array_type> velocity = particle_->velocity();
    cache_proxy<group_array_type const> group = group_->unordered();

    LOG_TRACE("first leapfrog half-step");

    scoped_timer_type timer(runtime_.integrate);

    for (size_type i : *group) {
        vector_type& v = (*velocity)[i];
        vector_type& r = (*position)[i];
        v += net_force_[i] * timestep_half_ / (*mass)[i];
        r += v * timestep_;
        (*image)[i] += box_->reduce_periodic(r);
    }
}

/**
 * Second leapfrog half-step of velocity-Verlet algorithm
 */
template <int dimension, typename float_type>
void verlet<dimension, float_type>::finalize()
{
    cache_proxy<mass_array_type const> mass = particle_->mass();
    cache_proxy<velocity_array_type> velocity = particle_->velocity();
    cache_proxy<group_array_type const> group = group_->unordered();

    LOG_TRACE("second leapfrog half-step");

    scoped_timer_type timer(runtime_.finalize);

    for (size_type i : *group) {
        (*velocity)[i] += net_force_[i] * timestep_half_ / (*mass)[i];
    }
}

template <int dimension, typename float_type>
void verlet<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("integrators")
            [
                class_<verlet>()
                    .def("acquire_net_force", &verlet::acquire_net_force)
                    .def("integrate", &verlet::integrate)
                    .def("finalize", &verlet::finalize)
                    .def("set_timestep", &verlet::set_timestep)
                    .property("timestep", &verlet::timestep)
                    .scope
                    [
                        class_<runtime>("runtime")
                            .def_readonly("integrate", &runtime::integrate)
                            .def_readonly("finalize", &runtime::finalize)
                    ]
                    .def_readonly("runtime", &verlet::runtime_)

              , def("verlet", &std::make_shared<verlet
                  , std::shared_ptr<particle_type>
                  , std::shared_ptr<particle_group_type>
                  , std::shared_ptr<force_type>
                  , std::shared_ptr<box_type const>
                  , double
                  , std::shared_ptr<logger_type>
                >)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_integrators_verlet(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    verlet<3, double>::luaopen(L);
    verlet<2, double>::luaopen(L);
#else
    verlet<3, float>::luaopen(L);
    verlet<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class verlet<3, double>;
template class verlet<2, double>;
#else
template class verlet<3, float>;
template class verlet<2, float>;
#endif

} // namespace mdsim
} // namespace host
} // namespace integrators
} // namespace halmd
