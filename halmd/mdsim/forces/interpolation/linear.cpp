/*
 * Copyright © 2013 Nicolas Höft
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

#include <halmd/mdsim/forces/interpolation/linear.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/demangle.hpp>

namespace halmd {
namespace mdsim {
namespace forces {
namespace interpolation {

template <int dimension, typename float_type>
linear<dimension, float_type>::linear(
    vector_type length
  , vector_type origin
  , index_type nknots
)
  : nknots_(nknots)
  , origin_(origin)
  , cell_length_(length)
{
    for (int d = 0; d < dimension; ++d) {
        // assume uniform grid and substract 1 because the number of
        // knots include both edges (ie. grid[dim][0] and grid[dim][nkots])
        grid_basis_[d] = cell_length_[d] / (nknots_[d] - 1);
    }
    total_knots_ = std::accumulate(nknots_.begin(), nknots_.end(), 1, std::multiplies<size_type>());
}

template <typename interpolation_type>
static std::function<typename interpolation_type::size_type ()>
wrap_total_knots(std::shared_ptr<interpolation_type> self)
{
    return [=]() {
        return self->total_knots();
    };
}

template <typename interpolation_type>
static std::function<typename interpolation_type::index_type ()>
wrap_nknots(std::shared_ptr<interpolation_type> self)
{
    return [=]() {
        return self->nknots();
    };
}

template <typename interpolation_type>
static std::function<typename interpolation_type::vector_type ()>
wrap_grid_basis(std::shared_ptr<interpolation_type> self)
{
    return [=]() {
        return self->grid_basis();
    };
}

template <int dimension, typename float_type>
void linear<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    static std::string class_name("linear_" + demangled_name<float_type>());
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("forces")
            [
                namespace_("interpolation")
                [
                    class_<linear>()
                        .property("total_knots", &wrap_total_knots<linear>)
                        .property("nknots", &wrap_nknots<linear>)
                        .property("grid_basis", &wrap_grid_basis<linear>)
                  , def(class_name.c_str(), &std::make_shared<linear
                        , vector_type
                        , vector_type
                        , index_type
                      >)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_forces_interpolation_linear(lua_State* L)
{
    linear<1, float>::luaopen(L);
    linear<2, float>::luaopen(L);
    linear<3, float>::luaopen(L);
    linear<1, double>::luaopen(L);
    linear<2, double>::luaopen(L);
    linear<3, double>::luaopen(L);
    return 0;
}

// explicit instantiation
template class linear<1, float>;
template class linear<2, float>;
template class linear<3, float>;
template class linear<1, double>;
template class linear<2, double>;
template class linear<3, double>;

} // namespace interpolation
} // namespace forces


} // namespace mdsim
} // namespace halmd