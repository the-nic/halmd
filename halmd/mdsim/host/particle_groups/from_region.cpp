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

#include <halmd/algorithm/host/radix_sort.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/mdsim/host/particle_groups/from_region.hpp>
#include <halmd/utility/lua/lua.hpp>

#include <algorithm>

namespace halmd {
namespace mdsim {
namespace host {
namespace particle_groups {

template <typename particle_type>
from_region<particle_type>::from_region(
    std::shared_ptr<particle_type const> particle
  , std::shared_ptr<region_type> region
  , selection selected_region
  , std::shared_ptr<logger> logger
)
  : particle_(particle)
  , region_(region)
  , selected_region_(selected_region)
  , logger_(logger)
{
}

template <typename particle_type>
cache<typename from_region<particle_type>::array_type> const&
from_region<particle_type>::ordered()
{
    auto const& mask_cache = region_->mask();
    if (mask_cache != ordered_cache_) {
        auto ordered = make_cache_mutable(ordered_);
        LOG_TRACE("ordered sequence of particle indices");
        auto it_range = (selected_region_ == excluded ? region_->excluded() : region_->included());
        ordered->clear(); // avoid copying the elements upon resize()
        ordered->resize(std::end(it_range) - std::begin(it_range));
        std::copy(std::begin(it_range), std::end(it_range), ordered->begin());

        ordered_cache_ = mask_cache;
    }
    return ordered_;
}

template <typename particle_type>
cache<typename from_region<particle_type>::array_type> const&
from_region<particle_type>::unordered()
{
    auto const& mask_cache = region_->mask();
    if (mask_cache != unordered_cache_) {
        auto unordered = make_cache_mutable(unordered_);
        LOG_TRACE("unordered sequence of particle indices");

        auto it_range = (selected_region_ == excluded ? region_->excluded() : region_->included());
        unordered->clear(); // avoid copying the elements upon resize()
        unordered->resize(std::end(it_range) - std::begin(it_range));
        std::copy(std::begin(it_range), std::end(it_range), unordered->begin());

        // TODO: is radix sort required here?
        radix_sort(
            unordered->begin()
          , unordered->end()
        );

        unordered_cache_ = mask_cache;
    }
    return unordered_;
}

template <typename particle_type>
cache<typename from_region<particle_type>::size_type> const&
from_region<particle_type>::size()
{
    auto size = make_cache_mutable(size_);
    *size = (selected_region_ == excluded ? region_->nexcluded() : region_->nincluded());
    return size_;
}

template <typename particle_group_type, typename particle_type>
static void
wrap_to_particle(std::shared_ptr<particle_group_type> self, std::shared_ptr<particle_type> particle_src, std::shared_ptr<particle_type> particle_dst)
{
    particle_group_to_particle(*particle_src, *self, *particle_dst);
}

template <typename particle_type>
void from_region<particle_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("particle_groups")
            [
                class_<from_region, particle_group>()
                    .def("to_particle", &wrap_to_particle<from_region<particle_type>, particle_type>)

              , def("from_region", &std::make_shared<from_region<particle_type>
                  , std::shared_ptr<particle_type const>
                  , std::shared_ptr<region_type>
                  , selection
                  , std::shared_ptr<logger>
                  >)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_particle_groups_from_region(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    from_region<particle<3, double>>::luaopen(L);
    from_region<particle<2, double>>::luaopen(L);
#else
    from_region<particle<3, float>>::luaopen(L);
    from_region<particle<2, float>>::luaopen(L);
#endif
    return 0;
}

} // namespace particle_groups
} // namespace host
} // namespace mdsim
} // namespace halmd
