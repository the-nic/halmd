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

#include <halmd/mdsim/host/random.hpp>
#include <halmd/util/logger.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace mdsim { namespace host
{

random::random(po::options const& vm)
  : _Base(vm)
{
    if (vm["random-seed"].empty()) {
        seed(readint(vm["random-device"].as<std::string>()));
    }
    else {
        seed(vm["random-seed"].as<unsigned int>());
    }
}

void random::seed(unsigned int value)
{
    LOG("random number generator seed: " << value);
    rng_.seed(value);
}

/**
 * Resolve module dependencies
 */
void random::resolve(po::options const& vm)
{
    // no dependencies
}

/**
 * Assemble module options
 */
void random::options(po::options_description& desc)
{
    desc.add_options()
        ("random-seed", po::value<unsigned int>(),
         "random number generator integer seed")
        ("random-device", po::value<std::string>()->default_value("/dev/random"),
         "random number generator device")
        ;
}

}} // namespace mdsim::host

template class module<mdsim::host::random>;

} // namespace halmd
