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

#include <algorithm>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/host/position/file.hpp>

namespace halmd
{
namespace mdsim { namespace host { namespace position
{

using namespace boost;
using namespace std;

/**
 * Resolve module dependencies
 */
template <int dimension, typename float_type>
void file<dimension, float_type>::depends()
{
    modules::depends<_Self, reader_type>::required();
    modules::depends<_Self, sample_type>::required();
    modules::depends<_Self, particle_type>::required();
    modules::depends<_Self, box_type>::required();
}

template <int dimension, typename float_type>
void file<dimension, float_type>::select(po::variables_map const& vm)
{
    if (vm["position"].as<string>() != "file") {
        throw unsuitable_module("mismatching option position");
    }
}

template <int dimension, typename float_type>
file<dimension, float_type>::file(modules::factory& factory, po::variables_map const& vm)
  : _Base(factory, vm)
  // dependency injection
  , reader(modules::fetch<reader_type>(factory, vm))
  , sample(modules::fetch<sample_type>(factory, vm))
  , particle(modules::fetch<particle_type>(factory, vm))
  , box(modules::fetch<box_type>(factory, vm))
{}

/**
 * set particle positions
 */
template <int dimension, typename float_type>
void file<dimension, float_type>::set()
{
    // assign particle coordinates
    for (size_t j = 0, i = 0; j < particle->ntype; i += particle->ntypes[j], ++j) {
        copy(sample->r[j]->begin(), sample->r[j]->end(), &particle->r[i]);
    }

    // shift particle positions to range (-L/2, L/2)
    for (size_t i = 0; i < particle->nbox; ++i) {
        box->reduce_periodic(particle->r[i]);
    }

    // assign particle image vectors
    fill(particle->image.begin(), particle->image.end(), 0);

    LOG("set particle positions from trajectory sample");
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class file<3, double>;
template class file<2, double>;
#else
template class file<3, float>;
template class file<2, float>;
#endif

}}} // namespace mdsim::host::position

#ifndef USE_HOST_SINGLE_PRECISION
template class module<mdsim::host::position::file<3, double> >;
template class module<mdsim::host::position::file<2, double> >;
#else
template class module<mdsim::host::position::file<3, float> >;
template class module<mdsim::host::position::file<2, float> >;
#endif

} // namespace halmd
