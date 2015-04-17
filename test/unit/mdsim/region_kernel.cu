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

#include <test/unit/mdsim/geometries/simple.hpp>

#include <halmd/mdsim/gpu/region_kernel.cu>

namespace halmd {
namespace mdsim {
namespace gpu {
template class geometry_predicate<simple_geometry<3, float> >;
template class geometry_predicate<simple_geometry<2, float> >;

template class region_wrapper<2, simple_geometry<2, float> >;
template class region_wrapper<3, simple_geometry<3, float> >;

}
}

/*template class algorithm::gpu::copy_if_wrapper<unsigned int, mdsim::gpu::geometry_predicate<simple_geometry<2, float> > >;
template class algorithm::gpu::copy_if_wrapper<unsigned int, mdsim::gpu::geometry_predicate<simple_geometry<3, float> > >;*/
}
