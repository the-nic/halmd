/*
 * Copyright © 2011  Peter Colberg
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

#include <boost/array.hpp>
#include <boost/function.hpp>
#include <luabind/luabind.hpp>
#include <vector>

#include <halmd/config.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/utility/lua/fixed_vector_converter.hpp>
#include <halmd/utility/lua/vector_converter.hpp>

using namespace boost;
using namespace std;

namespace halmd {

// This macro uses __VA_ARGS__ to support template types with commas.
// __VA_ARGS__ is part of the C99 standard, and will be part of C++0x,
// therefore most C++ compilers already support __VA_ARGS__ as an
// extension. This was tested with GCC 4.4 and Clang 2.9.
//
// The stringification turns the C++ type name into a Lua class name.
// A Lua class name may be any string of characters, e.g. spaces,
// commas, brackets or ampersands. As the registered classes are
// never constructed in Lua, but returned from C++ modules, the
// class names only have informational purposes. Use of the full
// C++ type name is especially useful for debugging argument
// mismatches, e.g. if the user tries to register an unsupported
// data slot with the H5MD writer. Luabind will then print all
// supported slot types, with the exact slot signatures.
//
#define slot(...)                                               \
    class_<__VA_ARGS__>(#__VA_ARGS__)                           \
        .def("__call", &__VA_ARGS__::operator())                \

/*
 * Lua bindings for boost::function<> with return value.
 *
 * This function registers all data slot types used in HALMD.
 * This allows retrieving a data slot from a C++ module in Lua, and
 * registering it with the H5MD or other writer, or running the slot
 * directly in Lua to convert the data to a Lua number or table.
 */
HALMD_LUA_API int luaopen_libhalmd_utility_lua_function(lua_State* L)
{
    using namespace luabind;
    module(L, "libhalmd")
    [
        slot( function<float ()> )
      , slot( function<float const& ()> )
      , slot( function<double ()> )
      , slot( function<double const& ()> )
      , slot( function<fixed_vector<float, 2> ()> )
      , slot( function<fixed_vector<float, 2> const& ()> )
      , slot( function<fixed_vector<float, 3> ()> )
      , slot( function<fixed_vector<float, 3> const& ()> )
      , slot( function<fixed_vector<double, 2> ()> )
      , slot( function<fixed_vector<double, 2> const& ()> )
      , slot( function<fixed_vector<double, 3> ()> )
      , slot( function<fixed_vector<double, 3> const& ()> )
      , slot( function<vector<float> ()> )
      , slot( function<vector<float> const& ()> )
      , slot( function<vector<double> ()> )
      , slot( function<vector<double> const& ()> )
      , slot( function<vector<fixed_vector<float, 2> > ()> )
      , slot( function<vector<fixed_vector<float, 2> > const& ()> )
      , slot( function<vector<fixed_vector<float, 3> > ()> )
      , slot( function<vector<fixed_vector<float, 3> > const& ()> )
      , slot( function<vector<fixed_vector<double, 2> > ()> )
      , slot( function<vector<fixed_vector<double, 2> > const& ()> )
      , slot( function<vector<fixed_vector<double, 3> > ()> )
      , slot( function<vector<fixed_vector<double, 3> > const& ()> )
      , slot( function<vector<array<float, 3> > ()> )
      , slot( function<vector<array<float, 3> > const& ()> )
      , slot( function<vector<array<double, 3> > ()> )
      , slot( function<vector<array<double, 3> > const& ()> )
      , slot( function<void (float&)> )
      , slot( function<void (double&)> )
      , slot( function<void (fixed_vector<float, 2>&)> )
      , slot( function<void (fixed_vector<float, 3>&)> )
      , slot( function<void (fixed_vector<double, 2>&)> )
      , slot( function<void (fixed_vector<double, 3>&)> )
      , slot( function<void (vector<float>&)> )
      , slot( function<void (vector<double>&)> )
      , slot( function<void (vector<fixed_vector<float, 2> >&)> )
      , slot( function<void (vector<fixed_vector<float, 3> >&)> )
      , slot( function<void (vector<fixed_vector<double, 2> >&)> )
      , slot( function<void (vector<fixed_vector<double, 3> >&)> )
      , slot( function<void (vector<array<float, 3> >&)> )
      , slot( function<void (vector<array<double, 3> >&)> )
    ];
    return 0;
}

} // namespace halmd
