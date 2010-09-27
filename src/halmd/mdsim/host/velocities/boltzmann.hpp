/*
 * Copyright © 2008-2010  Peter Colberg
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

#ifndef HALMD_MDSIM_HOST_VELOCITIES_BOLTZMANN_HPP
#define HALMD_MDSIM_HOST_VELOCITIES_BOLTZMANN_HPP

#include <utility>

#include <halmd/mdsim/host/particle.hpp>
#include <halmd/mdsim/host/velocity.hpp>
#include <halmd/random/host/random.hpp>
#include <halmd/options.hpp>

namespace halmd
{
namespace mdsim { namespace host { namespace velocities
{

template <int dimension, typename float_type>
class boltzmann
  : public host::velocity<dimension, float_type>
{
public:
    // module definitions
    typedef boltzmann _Self;
    typedef host::velocity<dimension, float_type> _Base;
    static void options(po::options_description& desc);
    static void depends();
    static void select(po::variables_map const& vm);

    typedef host::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef random::host::random random_type;

    shared_ptr<particle_type> particle;
    shared_ptr<random_type> random;

    boltzmann(modules::factory& factory, po::variables_map const& vm);
    virtual ~boltzmann() {};
    void set();

// private:
    /** assign new velocities from Gaussian distribution of width sigma,
      * return mean velocity and mean-square velocity */
    std::pair<vector_type, float_type> gaussian(float_type sigma);

protected:
    /** temperature */
    float_type temp_;
};

}}} // namespace mdsim::host::velocities

} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_VELOCITIES_BOLTZMANN_HPP */
