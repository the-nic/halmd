/*
 * Copyright © 2008-2012  Peter Colberg and Felix Höfling
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

#ifndef HALMD_MDSIM_GPU_PARTICLE_HPP
#define HALMD_MDSIM_GPU_PARTICLE_HPP

#include <algorithm>
#include <lua.hpp>
#include <vector>

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <halmd/mdsim/type_traits.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

template <int dimension, typename float_type>
class particle
{
public:
    typedef typename type_traits<dimension, float_type>::vector_type vector_type;
    typedef typename type_traits<dimension, float>::gpu::coalesced_vector_type gpu_vector_type;

    typedef vector_type position_type;
    typedef vector_type image_type;
    typedef vector_type velocity_type;
    typedef unsigned int tag_type;
    typedef unsigned int reverse_tag_type;
    typedef unsigned int species_type;
    typedef float mass_type;
    typedef vector_type force_type;
    typedef float_type en_pot_type;
    typedef typename type_traits<dimension, float_type>::stress_tensor_type stress_pot_type;
    typedef float_type hypervirial_type;

    typedef cuda::vector<float4> position_array_type;
    typedef cuda::vector<gpu_vector_type> image_array_type;
    typedef cuda::vector<float4> velocity_array_type;
    typedef cuda::vector<tag_type> tag_array_type;
    typedef cuda::vector<reverse_tag_type> reverse_tag_array_type;
    typedef cuda::vector<gpu_vector_type> force_array_type;
    typedef cuda::vector<en_pot_type> en_pot_array_type;
    typedef cuda::vector<typename type_traits<dimension, float_type>::gpu::stress_tensor_type> stress_pot_array_type;
    typedef cuda::vector<hypervirial_type> hypervirial_array_type;

    void set();
    void rearrange(cuda::vector<unsigned int> const& g_index);

    /** grid and block dimensions for CUDA calls */
    cuda::config const dim;

    /**
     * Allocate particle arrays in GPU memory.
     *
     * @param nparticle number of particles
     * @param nspecies number of particle species
     *
     * All particle arrays, except the masses, are initialised to zero.
     * The particle masses are initialised to unit mass.
     */
    particle(std::size_t nparticle, unsigned int nspecies = 1);

    /**
     * Returns number of particles.
     *
     * Currently the number of particles is fixed at construction of
     * particle. This may change in the future, to allow for chemical
     * reactions that do not conserve the number of particles, or to
     * transfer particles between domains of different processors.
     */
    std::size_t nparticle() const
    {
        return g_tag_.size();
    }

    /**
     * Returns number of particle placeholders.
     *
     * Currently the number of placeholders, i.e. the element count of the
     * particle arrays in memory, is equal to the total number of kernel
     * threads, which is a multiple of the number of threads per block,
     * and greater or equal than the number of particles.
     */
    std::size_t nplaceholder() const
    {
        return g_tag_.capacity();
    }

    /**
     * Returns number of species.
     */
    unsigned int nspecies() const
    {
        return nspecies_;
    }

    /**
     * Returns non-const reference to particle positions and species.
     */
    position_array_type const& position() const
    {
        return g_position_;
    }

    /**
     * Returns const reference to particle positions and species.
     */
    position_array_type& position()
    {
        return g_position_;
    }

    /**
     * Returns non-const reference to particle images.
     */
    image_array_type const& image() const
    {
        return g_image_;
    }

    /**
     * Returns const reference to particle images.
     */
    image_array_type& image()
    {
        return g_image_;
    }

    /**
     * Returns non-const reference to particle velocities and masses.
     */
    velocity_array_type const& velocity() const
    {
        return g_velocity_;
    }

    /**
     * Returns const reference to particle velocities and masses.
     */
    velocity_array_type& velocity()
    {
        return g_velocity_;
    }

    /**
     * Returns non-const reference to particle tags.
     */
    tag_array_type const& tag() const
    {
        return g_tag_;
    }

    /**
     * Returns const reference to particle tags.
     */
    tag_array_type& tag()
    {
        return g_tag_;
    }

    /**
     * Returns non-const reference to particle reverse tags.
     */
    reverse_tag_array_type const& reverse_tag() const
    {
        return g_reverse_tag_;
    }

    /**
     * Returns const reference to particle reverse tags.
     */
    reverse_tag_array_type& reverse_tag()
    {
        return g_reverse_tag_;
    }

    /**
     * Returns non-const reference to force per particle.
     */
    force_array_type const& force() const
    {
        return g_force_;
    }

    /**
     * Returns const reference to force per particle.
     */
    force_array_type& force()
    {
        return g_force_;
    }

    /**
     * Returns const reference to potential energy per particle.
     *
     * This method checks that the computation of auxiliary variables was enabled.
     */
    en_pot_array_type const& en_pot() const
    {
        assert_aux_valid();
        return g_en_pot_;
    }

    /**
     * Returns non-const reference to potential energy per particle.
     */
    en_pot_array_type& en_pot()
    {
        return g_en_pot_;
    }

    /**
     * Returns const reference to potential part of stress tensor per particle.
     *
     * This method checks that the computation of auxiliary variables was enabled.
     */
    stress_pot_array_type const& stress_pot() const
    {
        assert_aux_valid();
        return g_stress_pot_;
    }

    /**
     * Returns non-const reference to potential part of stress tensor per particle.
     */
    stress_pot_array_type& stress_pot()
    {
        return g_stress_pot_;
    }

    /**
     * Returns const reference to hypervirial per particle.
     *
     * This method checks that the computation of auxiliary variables was enabled.
     */
    hypervirial_array_type const& hypervirial() const
    {
        assert_aux_valid();
        return g_hypervirial_;
    }

    /**
     * Returns non-const reference to hypervirial per particle.
     */
    hypervirial_array_type& hypervirial()
    {
        return g_hypervirial_;
    }

    /**
     * Copy particle positions to given array.
     */
    template <typename iterator_type>
    iterator_type get_position(iterator_type const& first) const;

    /**
     * Copy particle positions from given array.
     */
    template <typename iterator_type>
    iterator_type set_position(iterator_type const& first);

    /**
     * Copy particle species to given array.
     */
    template <typename iterator_type>
    iterator_type get_species(iterator_type const& first) const;

    /**
     * Copy particle species from given array.
     */
    template <typename iterator_type>
    iterator_type set_species(iterator_type const& first);

    /**
     * Copy particle images to given array.
     */
    template <typename iterator_type>
    iterator_type get_image(iterator_type const& first) const;

    /**
     * Copy particle images from given array.
     */
    template <typename iterator_type>
    iterator_type set_image(iterator_type const& first);

    /**
     * Copy particle velocities to given array.
     */
    template <typename iterator_type>
    iterator_type get_velocity(iterator_type const& first) const;

    /**
     * Copy particle velocities from given array.
     */
    template <typename iterator_type>
    iterator_type set_velocity(iterator_type const& first);

    /**
     * Copy particle masses to given array.
     */
    template <typename iterator_type>
    iterator_type get_mass(iterator_type const& first) const;

    /**
     * Copy particle masses from given array.
     */
    template <typename iterator_type>
    iterator_type set_mass(iterator_type const& first);

    /**
     * Copy particle tags to given array.
     */
    template <typename iterator_type>
    iterator_type get_tag(iterator_type const& first) const;

    /**
     * Copy particle tags from given array.
     */
    template <typename iterator_type>
    iterator_type set_tag(iterator_type const& first);

    /**
     * Copy particle reverse tags to given array.
     */
    template <typename iterator_type>
    iterator_type get_reverse_tag(iterator_type const& first) const;

    /**
     * Copy particle reverse tags from given array.
     */
    template <typename iterator_type>
    iterator_type set_reverse_tag(iterator_type const& first);

    /**
     * Copy force per particle to given array.
     */
    template <typename iterator_type>
    iterator_type get_force(iterator_type const& first) const;

    /**
     * Copy force per particle from given array.
     */
    template <typename iterator_type>
    iterator_type set_force(iterator_type const& first);

    /**
     * Copy potential energy per particle to given array.
     */
    template <typename iterator_type>
    iterator_type get_en_pot(iterator_type const& first) const;

    /**
     * Copy potential energy per particle from given array.
     */
    template <typename iterator_type>
    iterator_type set_en_pot(iterator_type const& first);

    /**
     * Copy potential part of stress tensor per particle to given array.
     */
    template <typename iterator_type>
    iterator_type get_stress_pot(iterator_type const& first) const;

    /**
     * Copy potential part of stress tensor per particle from given array.
     */
    template <typename iterator_type>
    iterator_type set_stress_pot(iterator_type const& first);

    /**
     * Copy hypervirial per particle to given array.
     */
    template <typename iterator_type>
    iterator_type get_hypervirial(iterator_type const& first) const;

    /**
     * Copy hypervirial per particle from given array.
     */
    template <typename iterator_type>
    iterator_type set_hypervirial(iterator_type const& first);

    /**
     * Enable computation of auxiliary variables.
     *
     * The flag is reset by the next call to prepare().
     */
    void aux_enable();

    /**
     * Returns true if computation of auxiliary variables is enabled.
     */
    bool aux_valid() const
    {
        return aux_valid_;
    }

    /**
     * Reset forces, and optionally auxiliary variables, to zero.
     */
    void prepare();

    /**
     * Bind class to Lua.
     */
    static void luaopen(lua_State* L);

private:
    /** number of particle species */
    unsigned int nspecies_;
    /** positions, species */
    position_array_type g_position_;
    /** minimum image vectors */
    image_array_type g_image_;
    /** velocities, masses */
    velocity_array_type g_velocity_;
    /** particle tags */
    tag_array_type g_tag_;
    /** reverse particle tags */
    reverse_tag_array_type g_reverse_tag_;
    /** force per particle */
    force_array_type g_force_;
    /** potential energy per particle */
    en_pot_array_type g_en_pot_;
    /** potential part of stress tensor per particle */
    stress_pot_array_type g_stress_pot_;
    /** hypervirial per particle */
    hypervirial_array_type g_hypervirial_;

    /** flag for enabling the computation of auxiliary variables this step */
    bool aux_flag_;
    /** flag that indicates the auxiliary variables are computed this step */
    bool aux_valid_;

    void assert_aux_valid() const
    {
        if (!aux_valid_) {
            throw std::logic_error("auxiliary variables were not enabled in particle");
        }
    }

    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;
    typedef typename profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type rearrange;
    };

    /** profiling runtime accumulators */
    runtime runtime_;
};

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_position(iterator_type const& first) const
{
    cuda::host::vector<typename position_array_type::value_type> h_position(g_position_.size());
    cuda::copy(g_position_, h_position);
    iterator_type output = first;
    for (typename position_array_type::value_type const& v : h_position) {
        position_type position;
        species_type species;
        tie(position, species) <<= v;
        *output++ = position;
    }
    return output;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_position(iterator_type const& first)
{
    cuda::host::vector<typename position_array_type::value_type> h_position(g_position_.size());
    cuda::copy(g_position_, h_position);
    iterator_type input = first;
    for (typename position_array_type::value_type& v : h_position) {
        position_type position;
        species_type species;
        tie(position, species) <<= v;
        position = *input++;
        v <<= tie(position, species);
    }
#ifdef USE_VERLET_DSFUN
    cuda::memset(g_position_, 0, g_position_.capacity());
#endif
    cuda::copy(h_position, g_position_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_species(iterator_type const& first) const
{
    cuda::host::vector<typename position_array_type::value_type> h_position(g_position_.size());
    cuda::copy(g_position_, h_position);
    iterator_type output = first;
    for (typename position_array_type::value_type const& v : h_position) {
        position_type position;
        species_type species;
        tie(position, species) <<= v;
        *output++ = species;
    }
    return output;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_species(iterator_type const& first)
{
    cuda::host::vector<typename position_array_type::value_type> h_position(g_position_.size());
    cuda::copy(g_position_, h_position);
    iterator_type input = first;
    for (typename position_array_type::value_type& v : h_position) {
        position_type position;
        species_type species;
        tie(position, species) <<= v;
        species = *input++;
        v <<= tie(position, species);
    }
    cuda::copy(h_position, g_position_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_image(iterator_type const& first) const
{
    cuda::host::vector<typename image_array_type::value_type> h_image(g_image_.size());
    cuda::copy(g_image_, h_image);
    return std::copy(h_image.begin(), h_image.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_image(iterator_type const& first)
{
    cuda::host::vector<typename image_array_type::value_type> h_image(g_image_.size());
    iterator_type input = first;
    for (typename image_array_type::value_type& image : h_image) {
        image = *input++;
    }
    cuda::copy(h_image, g_image_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_velocity(iterator_type const& first) const
{
    cuda::host::vector<typename velocity_array_type::value_type> h_velocity(g_velocity_.size());
    cuda::copy(g_velocity_, h_velocity);
    iterator_type output = first;
    for (typename velocity_array_type::value_type const& v : h_velocity) {
        velocity_type velocity;
        mass_type mass;
        tie(velocity, mass) <<= v;
        *output++ = velocity;
    }
    return output;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_velocity(iterator_type const& first)
{
    cuda::host::vector<typename velocity_array_type::value_type> h_velocity(g_velocity_.size());
    cuda::copy(g_velocity_, h_velocity);
    iterator_type input = first;
    for (typename velocity_array_type::value_type& v : h_velocity) {
        velocity_type velocity;
        mass_type mass;
        tie(velocity, mass) <<= v;
        velocity = *input++;
        v <<= tie(velocity, mass);
    }
#ifdef USE_VERLET_DSFUN
    cuda::memset(g_velocity_, 0, g_velocity_.capacity());
#endif
    cuda::copy(h_velocity, g_velocity_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_mass(iterator_type const& first) const
{
    cuda::host::vector<typename velocity_array_type::value_type> h_velocity(g_velocity_.size());
    cuda::copy(g_velocity_, h_velocity);
    iterator_type output = first;
    for (typename velocity_array_type::value_type const& v : h_velocity) {
        velocity_type velocity;
        mass_type mass;
        tie(velocity, mass) <<= v;
        *output++ = mass;
    }
    return output;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_mass(iterator_type const& first)
{
    cuda::host::vector<typename velocity_array_type::value_type> h_velocity(g_velocity_.size());
    cuda::copy(g_velocity_, h_velocity);
    iterator_type input = first;
    for (typename velocity_array_type::value_type& v : h_velocity) {
        velocity_type velocity;
        mass_type mass;
        tie(velocity, mass) <<= v;
        mass = *input++;
        v <<= tie(velocity, mass);
    }
    cuda::copy(h_velocity, g_velocity_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_tag(iterator_type const& first) const
{
    cuda::host::vector<typename tag_array_type::value_type> h_tag(g_tag_.size());
    cuda::copy(g_tag_, h_tag);
    return std::copy(h_tag.begin(), h_tag.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_tag(iterator_type const& first)
{
    cuda::host::vector<typename tag_array_type::value_type> h_tag(g_tag_.size());
    iterator_type input = first;
    for (typename tag_array_type::value_type& tag : h_tag) {
        tag = *input++;
    }
    cuda::copy(h_tag, g_tag_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_reverse_tag(iterator_type const& first) const
{
    cuda::host::vector<typename reverse_tag_array_type::value_type> h_reverse_tag(g_reverse_tag_.size());
    cuda::copy(g_reverse_tag_, h_reverse_tag);
    return std::copy(h_reverse_tag.begin(), h_reverse_tag.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_reverse_tag(iterator_type const& first)
{
    cuda::host::vector<typename reverse_tag_array_type::value_type> h_reverse_tag(g_reverse_tag_.size());
    iterator_type input = first;
    for (typename reverse_tag_array_type::value_type& reverse_tag : h_reverse_tag) {
        reverse_tag = *input++;
    }
    cuda::copy(h_reverse_tag, g_reverse_tag_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_force(iterator_type const& first) const
{
    cuda::host::vector<typename force_array_type::value_type> h_force(g_force_.size());
    cuda::copy(g_force_, h_force);
    return std::copy(h_force.begin(), h_force.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_force(iterator_type const& first)
{
    cuda::host::vector<typename force_array_type::value_type> h_force(g_force_.size());
    iterator_type input = first;
    for (typename force_array_type::value_type& force : h_force) {
        force = *input++;
    }
    cuda::copy(h_force, g_force_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_en_pot(iterator_type const& first) const
{
    cuda::host::vector<typename en_pot_array_type::value_type> h_en_pot(g_en_pot_.size());
    cuda::copy(g_en_pot_, h_en_pot);
    return std::copy(h_en_pot.begin(), h_en_pot.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_en_pot(iterator_type const& first)
{
    cuda::host::vector<typename en_pot_array_type::value_type> h_en_pot(g_en_pot_.size());
    iterator_type input = first;
    for (typename en_pot_array_type::value_type& en_pot : h_en_pot) {
        en_pot = *input++;
    }
    cuda::copy(h_en_pot, g_en_pot_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_stress_pot(iterator_type const& first) const
{
    cuda::host::vector<typename stress_pot_array_type::value_type> h_stress_pot(g_stress_pot_.size());
    cuda::copy(g_stress_pot_, h_stress_pot);
    return std::copy(h_stress_pot.begin(), h_stress_pot.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_stress_pot(iterator_type const& first)
{
    cuda::host::vector<typename stress_pot_array_type::value_type> h_stress_pot(g_stress_pot_.size());
    iterator_type input = first;
    for (typename stress_pot_array_type::value_type& stress_pot : h_stress_pot) {
        stress_pot = *input++;
    }
    cuda::copy(h_stress_pot, g_stress_pot_);
    return input;
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::get_hypervirial(iterator_type const& first) const
{
    cuda::host::vector<typename hypervirial_array_type::value_type> h_hypervirial(g_hypervirial_.size());
    cuda::copy(g_hypervirial_, h_hypervirial);
    return std::copy(h_hypervirial.begin(), h_hypervirial.end(), first);
}

template <int dimension, typename float_type>
template <typename iterator_type>
inline iterator_type particle<dimension, float_type>::set_hypervirial(iterator_type const& first)
{
    cuda::host::vector<typename hypervirial_array_type::value_type> h_hypervirial(g_hypervirial_.size());
    iterator_type input = first;
    for (typename hypervirial_array_type::value_type& hypervirial : h_hypervirial) {
        hypervirial = *input++;
    }
    cuda::copy(h_hypervirial, g_hypervirial_);
    return input;
}

} // namespace mdsim
} // namespace gpu
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_PARTICLE_HPP */
