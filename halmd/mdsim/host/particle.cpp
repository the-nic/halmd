/*
 * Copyright © 2008-2012 Peter Colberg
 * Copyright © 2010-2012 Felix Höfling
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

#include <halmd/config.hpp>

#include <algorithm>
#include <boost/function_output_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <exception>
#include <iterator> // std::back_inserter
#include <luaponte/luaponte.hpp>
#include <luaponte/out_value_policy.hpp>
#include <numeric>

#include <halmd/algorithm/host/permute.hpp>
#include <halmd/io/logger.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/signal.hpp>

namespace halmd {
namespace mdsim {
namespace host {

template <int dimension, typename float_type>
particle<dimension, float_type>::particle(size_t nparticle, unsigned int nspecies)
  // allocate particle storage
  : nspecies_(std::max(nspecies, 1u))
  , position_(nparticle)
  , image_(nparticle)
  , velocity_(nparticle)
  , tag_(nparticle)
  , reverse_tag_(nparticle)
  , species_(nparticle)
  , mass_(nparticle)
  , force_(nparticle)
  , en_pot_(nparticle)
  , stress_pot_(nparticle)
  , hypervirial_(nparticle)
  // disable auxiliary variables by default
  , aux_flag_(false)
  , aux_valid_(false)
{
    cache_proxy<reverse_tag_array_type> reverse_tag = reverse_tag_;

    std::fill(position_.begin(), position_.end(), 0);
    std::fill(image_.begin(), image_.end(), 0);
    std::fill(velocity_.begin(), velocity_.end(), 0);
    std::iota(tag_.begin(), tag_.end(), 0);
    std::iota(reverse_tag->begin(), reverse_tag->end(), 0);
    std::fill(species_.begin(), species_.end(), 0);
    std::fill(mass_.begin(), mass_.end(), 1);
    std::fill(force_.begin(), force_.end(), 0);
    std::fill(en_pot_.begin(), en_pot_.end(), 0);
    std::fill(stress_pot_.begin(), stress_pot_.end(), 0);
    std::fill(hypervirial_.begin(), hypervirial_.end(), 0);

    LOG("number of particles: " << tag_.size());
    LOG("number of particle species: " << nspecies_);
}

template <int dimension, typename float_type>
void particle<dimension, float_type>::aux_enable()
{
    LOG_TRACE("enable computation of auxiliary variables");
    aux_flag_ = true;
}

template <int dimension, typename float_type>
void particle<dimension, float_type>::prepare()
{
    LOG_TRACE("zero forces");
    std::fill(force_.begin(), force_.end(), 0);

    // indicate whether auxiliary variables are computed this step
    aux_valid_ = aux_flag_;

    if (aux_flag_) {
        LOG_TRACE("zero auxiliary variables");
        std::fill(en_pot_.begin(), en_pot_.end(), 0);
        std::fill(stress_pot_.begin(), stress_pot_.end(), 0);
        std::fill(hypervirial_.begin(), hypervirial_.end(), 0);
        aux_flag_ = false;
    }
}

/**
 * Rearrange particles in memory according to an integer index sequence
 *
 * The neighbour lists must be rebuilt after calling this function!
 */
template <int dimension, typename float_type>
void particle<dimension, float_type>::rearrange(std::vector<unsigned int> const& index)
{
    scoped_timer_type timer(runtime_.rearrange);

    cache_proxy<reverse_tag_array_type> reverse_tag = reverse_tag_;

    permute(position_.begin(), position_.end(), index.begin());
    permute(image_.begin(), image_.end(), index.begin());
    permute(velocity_.begin(), velocity_.end(), index.begin());
    // no permutation of forces
    permute(tag_.begin(), tag_.end(), index.begin());
    permute(species_.begin(), species_.end(), index.begin());
    permute(mass_.begin(), mass_.end(), index.begin());

    // update reverse tags
    for (unsigned int i = 0; i < tag_.size(); ++i) {
        assert(tag_[i] < reverse_tag->size());
        (*reverse_tag)[tag_[i]] = i;
    }
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::position_type> ()>
wrap_get_position(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::position_type> {
        std::vector<typename particle_type::position_type> output;
        output.reserve(self->nparticle());
        get_position(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::position_type> const&)>
wrap_set_position(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::position_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_position(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::image_type> ()>
wrap_get_image(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::image_type> {
        std::vector<typename particle_type::image_type> output;
        output.reserve(self->nparticle());
        get_image(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::image_type> const&)>
wrap_set_image(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::image_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_image(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::velocity_type> ()>
wrap_get_velocity(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::velocity_type> {
        std::vector<typename particle_type::velocity_type> output;
        output.reserve(self->nparticle());
        get_velocity(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::velocity_type> const&)>
wrap_set_velocity(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::velocity_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_velocity(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::tag_type> ()>
wrap_get_tag(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::tag_type> {
        std::vector<typename particle_type::tag_type> output;
        output.reserve(self->nparticle());
        get_tag(
            *self
          , boost::make_function_output_iterator([&](typename particle_type::tag_type t) {
                output.push_back(t + 1);
            })
        );
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::tag_type> const&)>
wrap_set_tag(std::shared_ptr<particle_type> self)
{
    typedef typename particle_type::tag_type tag_type;
    return [=](std::vector<tag_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        tag_type nparticle = self->nparticle();
        set_tag(
            *self
          , boost::make_transform_iterator(input.begin(), [&](tag_type t) -> tag_type {
                if (t < 1 || t > nparticle) {
                    throw std::invalid_argument("invalid particle tag");
                }
                return t - 1;
            })
        );
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::reverse_tag_type> ()>
wrap_get_reverse_tag(std::shared_ptr<particle_type const> self)
{
    typedef typename particle_type::reverse_tag_type reverse_tag_type;
    return [=]() -> std::vector<reverse_tag_type> {
        std::vector<typename particle_type::reverse_tag_type> output;
        output.reserve(self->nparticle());
        get_reverse_tag(
            *self
          , boost::make_function_output_iterator([&](reverse_tag_type i) {
                output.push_back(i + 1);
            })
        );
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::reverse_tag_type> const&)>
wrap_set_reverse_tag(std::shared_ptr<particle_type> self)
{
    typedef typename particle_type::reverse_tag_type reverse_tag_type;
    return [=](std::vector<reverse_tag_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        reverse_tag_type nparticle = self->nparticle();
        set_reverse_tag(
            *self
          , boost::make_transform_iterator(input.begin(), [&](reverse_tag_type i) -> reverse_tag_type {
                if (i < 1 || i > nparticle) {
                    throw std::invalid_argument("invalid particle reverse tag");
                }
                return i - 1;
            })
        );
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::species_type> ()>
wrap_get_species(std::shared_ptr<particle_type const> self)
{
    typedef typename particle_type::species_type species_type;
    return [=]() -> std::vector<species_type> {
        std::vector<species_type> output;
        output.reserve(self->nparticle());
        get_species(
            *self
          , boost::make_function_output_iterator([&](typename particle_type::species_type s) {
                output.push_back(s + 1);
            })
        );
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::species_type> const&)>
wrap_set_species(std::shared_ptr<particle_type> self)
{
    typedef typename particle_type::species_type species_type;
    return [=](std::vector<species_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        species_type nspecies = self->nspecies();
        set_species(
            *self
          , boost::make_transform_iterator(input.begin(), [&](species_type s) -> species_type {
                if (s < 1 || s > nspecies) {
                    throw std::invalid_argument("invalid particle species");
                }
                return s - 1;
            })
        );
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::mass_type> ()>
wrap_get_mass(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::mass_type> {
        std::vector<typename particle_type::mass_type> output;
        output.reserve(self->nparticle());
        get_mass(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::mass_type> const&)>
wrap_set_mass(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::mass_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_mass(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::force_type> ()>
wrap_get_force(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::force_type> {
        std::vector<typename particle_type::force_type> output;
        output.reserve(self->nparticle());
        get_force(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::force_type> const&)>
wrap_set_force(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::force_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_force(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::en_pot_type> ()>
wrap_get_en_pot(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::en_pot_type> {
        std::vector<typename particle_type::en_pot_type> output;
        output.reserve(self->nparticle());
        get_en_pot(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::en_pot_type> const&)>
wrap_set_en_pot(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::en_pot_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_en_pot(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::stress_pot_type> ()>
wrap_get_stress_pot(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::stress_pot_type> {
        std::vector<typename particle_type::stress_pot_type> output;
        output.reserve(self->nparticle());
        get_stress_pot(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::stress_pot_type> const&)>
wrap_set_stress_pot(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::stress_pot_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_stress_pot(*self, input.begin());
    };
}

template <typename particle_type>
static std::function<std::vector<typename particle_type::hypervirial_type> ()>
wrap_get_hypervirial(std::shared_ptr<particle_type const> self)
{
    return [=]() -> std::vector<typename particle_type::hypervirial_type> {
        std::vector<typename particle_type::hypervirial_type> output;
        output.reserve(self->nparticle());
        get_hypervirial(*self, back_inserter(output));
        return std::move(output);
    };
}

template <typename particle_type>
static std::function<void (std::vector<typename particle_type::hypervirial_type> const&)>
wrap_set_hypervirial(std::shared_ptr<particle_type> self)
{
    return [=](std::vector<typename particle_type::hypervirial_type> const& input) {
        if (input.size() != self->nparticle()) {
            throw std::invalid_argument("input array size not equal to number of particles");
        }
        set_hypervirial(*self, input.begin());
    };
}

template <int dimension, typename float_type>
static int wrap_dimension(particle<dimension, float_type> const&)
{
    return dimension;
}

template <typename particle_type>
static std::function<void ()>
wrap_aux_enable(std::shared_ptr<particle_type> self)
{
    return [=]() {
        self->aux_enable();
    };
}

template <typename particle_type>
static std::function<void ()>
wrap_prepare(std::shared_ptr<particle_type> self)
{
    return [=]() {
        self->prepare();
    };
}

template <typename particle_type>
struct wrap_particle
  : particle_type
  , luaponte::wrap_base
{
    wrap_particle(size_t nparticle, unsigned int nspecies) : particle_type(nparticle, nspecies) {}
};

template <int dimension, typename float_type>
void particle<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    static std::string class_name = "particle_" + std::to_string(dimension);
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                class_<particle, std::shared_ptr<particle>, wrap_particle<particle> >(class_name.c_str())
                    .def(constructor<size_t, unsigned int>())
                    .property("nparticle", &particle::nparticle)
                    .property("nspecies", &particle::nspecies)
                    .property("get_position", &wrap_get_position<particle>)
                    .property("set_position", &wrap_set_position<particle>)
                    .property("get_image", &wrap_get_image<particle>)
                    .property("set_image", &wrap_set_image<particle>)
                    .property("get_velocity", &wrap_get_velocity<particle>)
                    .property("set_velocity", &wrap_set_velocity<particle>)
                    .property("get_tag", &wrap_get_tag<particle>)
                    .property("set_tag", &wrap_set_tag<particle>)
                    .property("get_reverse_tag", &wrap_get_reverse_tag<particle>)
                    .property("set_reverse_tag", &wrap_set_reverse_tag<particle>)
                    .property("get_species", &wrap_get_species<particle>)
                    .property("set_species", &wrap_set_species<particle>)
                    .property("get_mass", &wrap_get_mass<particle>)
                    .property("set_mass", &wrap_set_mass<particle>)
                    .property("get_force", &wrap_get_force<particle>)
                    .property("set_force", &wrap_set_force<particle>)
                    .property("get_en_pot", &wrap_get_en_pot<particle>)
                    .property("set_en_pot", &wrap_set_en_pot<particle>)
                    .property("get_stress_pot", &wrap_get_stress_pot<particle>)
                    .property("set_stress_pot", &wrap_set_stress_pot<particle>)
                    .property("get_hypervirial", &wrap_get_hypervirial<particle>)
                    .property("set_hypervirial", &wrap_set_hypervirial<particle>)
                    .property("dimension", &wrap_dimension<dimension, float_type>)
                    .property("aux_enable", &wrap_aux_enable<particle>)
                    .property("prepare", &wrap_prepare<particle>)
                    .scope[
                        class_<runtime>("runtime")
                            .def_readonly("rearrange", &runtime::rearrange)
                    ]
                    .def_readonly("runtime", &particle::runtime_)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_particle(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    particle<3, double>::luaopen(L);
    particle<2, double>::luaopen(L);
#else
    particle<3, float>::luaopen(L);
    particle<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class particle<3, double>;
template class particle<2, double>;
#else
template class particle<3, float>;
template class particle<2, float>;
#endif

} // namespace host
} // namespace mdsim
} // namespace halmd
