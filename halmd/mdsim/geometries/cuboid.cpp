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

#include <string>

#include <halmd/mdsim/geometries/cuboid.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/demangle.hpp>

namespace halmd {
namespace mdsim {
namespace geometries {

template <int dimension, typename float_type>
cuboid<dimension, float_type>::cuboid(vector_type origin, vector_type length)
  : origin_(origin)
  , edge_length_(length)
{
}

template <int dimension, typename float_type>
void cuboid<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    static std::string class_name("cuboid_" + demangled_name<float_type>());
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("geometries")
            [
                class_<cuboid>()
              , def(class_name.c_str(), &std::make_shared<cuboid
                  , vector_type
                  , vector_type
                  >)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_geometries_cuboid(lua_State* L)
{
    cuboid<2, float>::luaopen(L);
    cuboid<2, double>::luaopen(L);
    cuboid<3, float>::luaopen(L);
    cuboid<3, double>::luaopen(L);
    return 0;
}

// explicit instantiation
template class cuboid<2, float>;
template class cuboid<2, double>;
template class cuboid<3, float>;
template class cuboid<3, double>;

} // namespace geometries
} // namespace mdsim
} // namespace halmd
