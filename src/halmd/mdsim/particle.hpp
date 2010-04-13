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

#ifndef HALMD_MDSIM_PARTICLE_HPP
#define HALMD_MDSIM_PARTICLE_HPP

#include <vector>

#include <halmd/mdsim/factory.hpp>
#include <halmd/options.hpp>

namespace halmd { namespace mdsim
{

template <int dimension, typename float_type>
class particle
{
public:
    particle(options const& vm);
    virtual ~particle() {}

    /** number of particles in simulation box */
    unsigned int nbox;
    /** number of particle types */
    unsigned int ntype;
};

template <int dimension, typename float_type>
class factory<particle<dimension, float_type> >
{
public:
    typedef boost::shared_ptr<particle<dimension, float_type> > particle_ptr;
    static particle_ptr fetch(options const& vm);

private:
    static particle_ptr particle_;
};

}} // namespace halmd::mdsim

#endif /* ! HALMD_MDSIM_PARTICLE_HPP */
