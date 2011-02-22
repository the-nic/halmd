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

#include <halmd/observables/gpu/samples/trajectory.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace observables { namespace gpu { namespace samples
{

template <int dimension, typename float_type>
trajectory<dimension, float_type>::trajectory(vector<unsigned int> ntypes)
  // allocate sample pointers
  : r(ntypes.size())
  , v(ntypes.size())
  , time(-numeric_limits<double>::epsilon()) //< any value < 0.
{
    for (size_t i = 0; i < ntypes.size(); ++i) {
        r[i].reset(new sample_vector(ntypes[i]));
        v[i].reset(new sample_vector(ntypes[i]));
    }
}

template <int dimension, typename float_type>
void trajectory<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("trajectory_" + lexical_cast<string>(dimension) + "_");
    module(L, "halmd_wrapper")
    [
        namespace_("observables")
        [
            namespace_("gpu")
            [
                namespace_("samples")
                [
                    class_<trajectory, shared_ptr<trajectory> >(class_name.c_str())
                        .def(constructor<vector<unsigned int> >())
                ]
            ]
        ]
    ];
}

namespace // limit symbols to translation unit
{

__attribute__((constructor)) void register_lua()
{
    lua_wrapper::register_(0) //< distance of derived to base class
    [
        &trajectory<3, float>::luaopen
    ]
    [
        &trajectory<2, float>::luaopen
    ];
}

} // namespace

template class trajectory<3, float>;
template class trajectory<2, float>;

}}} // namespace observables::gpu::samples

} // namespace halmd
