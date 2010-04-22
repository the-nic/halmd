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

#ifndef HALMD_MDSIM_SAMPLES_HOST_TRAJECTORY_HPP
#define HALMD_MDSIM_SAMPLES_HOST_TRAJECTORY_HPP

#include <boost/shared_ptr.hpp>
#include <vector>

#include <halmd/utility/module.hpp>
#include <halmd/options.hpp>

namespace halmd { namespace mdsim { namespace samples { namespace host
{

template <int dimension, typename float_type>
class trajectory;

template <int dimension>
class trajectory<dimension, double>
{
public:
    trajectory(options const& vm) {}
    virtual ~trajectory() {}
    virtual void acquire() = 0;

    typedef typename module<trajectory>::pointer pointer;
    static pointer create(options const& vm);

    /** sample vector types for single particle */
    typedef numeric::host::blas::vector<double, dimension> position_vector;
    typedef numeric::host::blas::vector<double, dimension> velocity_vector;
    /** sample vector types for all particles of a species */
    typedef std::vector<position_vector> position_sample_vector;
    typedef std::vector<velocity_vector> velocity_sample_vector;
    /** sample pointer types for all particle of a species */
    typedef boost::shared_ptr<position_sample_vector> position_sample_pointer;
    typedef boost::shared_ptr<velocity_sample_vector> velocity_sample_pointer;
    /** sample pointer types for all species */
    typedef std::vector<position_sample_pointer> position_sample_pointer_vector;
    typedef std::vector<velocity_sample_pointer> velocity_sample_pointer_vector;

    /** periodically extended particle positions */
    position_sample_pointer_vector r;
    /** particle velocities */
    velocity_sample_pointer_vector v;
};

template <int dimension>
class trajectory<dimension, float>
{
public:
    trajectory(options const& vm) {}
    virtual ~trajectory() {}
    virtual void acquire() = 0;

    typedef typename module<trajectory>::pointer pointer;
    static pointer create(options const& vm);

    /** sample vector types for single particle */
    typedef numeric::host::blas::vector<float, dimension> position_vector;
    typedef numeric::host::blas::vector<float, dimension> velocity_vector;
    /** sample vector types for all particles of a species */
    typedef std::vector<position_vector> position_sample_vector;
    typedef std::vector<velocity_vector> velocity_sample_vector;
    /** sample pointer types for all particle of a species */
    typedef boost::shared_ptr<position_sample_vector> position_sample_pointer;
    typedef boost::shared_ptr<velocity_sample_vector> velocity_sample_pointer;
    /** sample pointer types for all species */
    typedef std::vector<position_sample_pointer> position_sample_pointer_vector;
    typedef std::vector<velocity_sample_pointer> velocity_sample_pointer_vector;

    /** periodically extended particle positions */
    position_sample_pointer_vector r;
    /** particle velocities */
    velocity_sample_pointer_vector v;
};

}}}} // namespace halmd::mdsim::samples::host

#endif /* ! HALMD_MDSIM_SAMPLES_HOST_TRAJECTORY_HPP */
