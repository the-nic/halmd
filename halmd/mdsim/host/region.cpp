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

#include <halmd/mdsim/host/region.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/algorithm/host/radix_sort.hpp>

#include <halmd/mdsim/geometries/cuboid.hpp>

#include <boost/iterator/counting_iterator.hpp>

namespace halmd {
namespace mdsim {
namespace host {

/**
 * construct region module
 *
 * @param particle mdsim::gpu::particle instance
 * @param box mdsim::box instance
 */
template <int dimension, typename float_type, typename geometry_type>
region<dimension, float_type, geometry_type>::region(
    std::shared_ptr<particle_type const> particle
  , std::shared_ptr<box_type const> box
  , std::shared_ptr<geometry_type const> geometry
  , std::shared_ptr<logger> logger
)
  : particle_(particle)
  , box_(box)
  , logger_(logger)
  , geometry_(geometry)
  , mask_(particle->nparticle())
{
}

template <int dimension, typename float_type, typename geometry_type>
typename region<dimension, float_type, geometry_type>::iterator_range_type
region<dimension, float_type, geometry_type>::excluded()
{
    update_();
    return boost::make_iterator_range(excluded_particles_.begin(), excluded_particles_.end());
}

template <int dimension, typename float_type, typename geometry_type>
typename region<dimension, float_type, geometry_type>::iterator_range_type
region<dimension, float_type, geometry_type>::included()
{
    update_();
    return boost::make_iterator_range(included_particles_.begin(), included_particles_.end());
}

template <int dimension, typename float_type, typename geometry_type>
cache<typename region<dimension, float_type, geometry_type>::array_type> const&
region<dimension, float_type, geometry_type>::mask()
{
    update_();
    return mask_;
}

/**
 * update the particle lists for the region
 */
template <int dimension, typename float_type, typename geometry_type>
void region<dimension, float_type, geometry_type>::update_()
{
    cache<position_array_type> const& position_cache = particle_->position();
    if (position_cache != mask_cache_) {
        LOG_TRACE("update region");

        scoped_timer_type timer(runtime_.update_mask);

        auto mask = make_cache_mutable(mask_);
        position_array_type const& position = read_cache(particle_->position());

        included_particles_.clear();
        excluded_particles_.clear();

        for (size_type i = 0; i < particle_->nparticle(); ++i) {

            vector_type r = position[i];
            box_->reduce_periodic(r);
            bool included  = (*geometry_)(r);
            (*mask)[i] = included ? 1 : 0;
            if (!included) {
                excluded_particles_.push_back(i);
            }
            else {
                included_particles_.push_back(i);
            }
        }
        mask_cache_ = position_cache;
    }
}

template <int dimension, typename float_type, typename geometry_type>
void region<dimension, float_type, geometry_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            class_<region, region_base>()
                .scope
                [
                    class_<runtime>("runtime")
                        .def_readonly("update_mask", &runtime::update_mask)
                        .def_readonly("update_permutation", &runtime::update_permutation)
                ]
                .def_readonly("runtime", &region::runtime_)
          , def("region", &std::make_shared<region
                  , std::shared_ptr<particle_type const>
                  , std::shared_ptr<box_type const>
                  , std::shared_ptr<geometry_type const>
              >)
        ]
    ];
}

void region_base::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            class_<region_base>()
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_region(lua_State* L)
{
    region_base::luaopen(L);
#ifndef USE_HOST_SINGLE_PRECISION
    region<3, double, mdsim::geometries::cuboid<3, double>>::luaopen(L);
    region<2, double, mdsim::geometries::cuboid<2, double>>::luaopen(L);
#else
    region<3, float, mdsim::geometries::cuboid<3, float>>::luaopen(L);
    region<2, float, mdsim::geometries::cuboid<2, float>>::luaopen(L);
#endif
    return 0;
}

} // namespace host
} // namespace mdsim
} // namespace halmd
