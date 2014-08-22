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

#include <exception>

#include <halmd/mdsim/gpu/region.hpp>
#include <halmd/algorithm/gpu/radix_sort.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

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
{
    try {
        auto mask = make_cache_mutable(mask_);
        mask->reserve(particle_->dim.threads());
        mask->resize(particle_->nparticle());
        g_particle_permutation_.reserve(particle_->dim.threads());
        g_particle_permutation_.resize(particle_->nparticle());
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to allocate global device memory");
        throw;
    }
}

template <int dimension, typename float_type, typename geometry_type>
typename region<dimension, float_type, geometry_type>::iterator_range_type
region<dimension, float_type, geometry_type>::excluded()
{
    update_permutation_();
    return boost::make_iterator_range(g_particle_permutation_.begin(), g_particle_permutation_.begin() + nexcluded_);
}

template <int dimension, typename float_type, typename geometry_type>
typename region<dimension, float_type, geometry_type>::iterator_range_type
region<dimension, float_type, geometry_type>::included()
{
    update_permutation_();
    return boost::make_iterator_range(g_particle_permutation_.begin() + nexcluded_, g_particle_permutation_.end());
}

template <int dimension, typename float_type, typename geometry_type>
cache<typename region<dimension, float_type, geometry_type>::array_type> const&
region<dimension, float_type, geometry_type>::mask()
{
    update_mask_();
    return mask_;
}

/**
 * update the particle lists for the region
 */
template <int dimension, typename float_type, typename geometry_type>
void region<dimension, float_type, geometry_type>::update_mask_()
{
    cache<position_array_type> const& position_cache = particle_->position();

    if (position_cache != mask_cache_) {
        scoped_timer_type timer(runtime_.update_mask);

        auto mask = make_cache_mutable(mask_);
        position_array_type const& position = read_cache(particle_->position());
        auto const* kernel = &region_wrapper<dimension, geometry_type>::kernel;
        // calculate "bin", ie. inside/outside the region
        cuda::memset(*mask, 0xFF);
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        kernel->compute_mask(
            &*position.begin()
          , particle_->nparticle()
          , &*mask->begin()
          , *geometry_
          , static_cast<position_type>(box_->length())
        );
        mask_cache_ = position_cache;
    }
}

/**
 * update the particle lists for the region
 */
template <int dimension, typename float_type, typename geometry_type>
void region<dimension, float_type, geometry_type>::update_permutation_()
{
    update_mask_();

    cache<position_array_type> const& position_cache = particle_->position();
    if(position_cache != permutation_cache_) {
        scoped_timer_type timer(runtime_.update_permutation);

        unsigned int nparticle = particle_->nparticle();
        auto const& mask = read_cache(mask_);

        // make a copy of the unordered mask that will be used for ordering
        array_type g_ordered_mask(mask.size());
        g_ordered_mask.reserve(mask.capacity());
        cuda::copy(mask.begin(), mask.end(), g_ordered_mask.begin());

        // generate particle permutation ordered by excluded/included
        auto const* kernel = &region_wrapper<dimension, geometry_type>::kernel;
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        kernel->gen_index(g_particle_permutation_, nparticle);
        radix_sort(g_ordered_mask.begin(), g_ordered_mask.end(), g_particle_permutation_.begin());

        cuda::vector<unsigned int> g_offset(1);
        cuda::host::vector<unsigned int> h_offset(1);
        cuda::memset(g_offset, 0xFF);
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        kernel->compute_bin_border(g_offset, g_ordered_mask, nparticle);
        cuda::copy(g_offset, h_offset);
        // if the first element of the mask array is -1, this means
        // the offset has not been set and we need to evaluate the first
        // element of the mask in order to see if all or none particles
        // are within the region
        if (h_offset.front() == static_cast<unsigned int>(-1)) {
            cuda::host::vector<unsigned int> h_first_mask(1);
            cuda::copy(g_ordered_mask.begin(), g_ordered_mask.begin() + 1, h_first_mask.begin());
            if (h_first_mask.front() == 1) {
                nexcluded_ = 0;
            }
            else {
                nexcluded_ = nparticle;
            }
        } else {
            nexcluded_ = h_offset.front();
        }
        permutation_cache_ = position_cache;
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

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_region(lua_State* L)
{
    region_base::luaopen(L);
    return 0;
}

} // namespace gpu
} // namepsace mdsim
} // namespace halmd
