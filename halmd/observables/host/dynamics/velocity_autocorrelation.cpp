/*
 * Copyright © 2008-2011  Peter Colberg and Felix Höfling
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

#include <halmd/io/logger.hpp>
#include <halmd/observables/dynamics/correlation.hpp>
#include <halmd/observables/host/dynamics/velocity_autocorrelation.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace observables {
namespace host {
namespace dynamics {

template <int dimension, typename float_type>
velocity_autocorrelation<dimension, float_type>::velocity_autocorrelation(
    size_t type
)
  // member initialisation
  : type_(type)
{
    LOG("initialise velocity autocorrelation of " << string(1, 'A' + type) << " particles");
}

/**
 * Compute velocity autocorrelation of two velocity sample vectors.
 *
 * @param first particles velocities of one species at time t1
 * @param second particles velocities of one species at time t2
 * @returns accumulated velocity autocorrelation
 */
template <int dimension, typename float_type>
typename velocity_autocorrelation<dimension, float_type>::accumulator_type
velocity_autocorrelation<dimension, float_type>::compute(
    sample_type const& first
  , sample_type const& second
)
{
    accumulator_type acc;
    typename sample_type::sample_vector::const_iterator v1, v2, end = first.v[type_]->end();
    for (v1 = first.v[type_]->begin(), v2 = second.v[type_]->begin(); v1 != end; ++v1, ++v2) {
        // accumulate velocity autocorrelation
        acc(correlate_function_type()(*v1, *v2));
    }
    return acc;
}

template <typename tcf_type>
static boost::shared_ptr<tcf_type>
wrap_tcf(size_t type, typename tcf_type::sample_type const&)
{
    return make_shared<tcf_type>(type);
}

template <int dimension, typename float_type>
void velocity_autocorrelation<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("velocity_autocorrelation_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("observables")
        [
            namespace_("dynamics")
            [
                class_<velocity_autocorrelation>(class_name.c_str())

              , def("velocity_autocorrelation", &wrap_tcf<velocity_autocorrelation>)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_observables_host_dynamics_velocity_autocorrelation(lua_State* L)
{
    velocity_autocorrelation<3, double>::luaopen(L);
    velocity_autocorrelation<2, double>::luaopen(L);
    velocity_autocorrelation<3, float>::luaopen(L);
    velocity_autocorrelation<2, float>::luaopen(L);
    observables::dynamics::correlation<velocity_autocorrelation<3, double> >::luaopen(L);
    observables::dynamics::correlation<velocity_autocorrelation<2, double> >::luaopen(L);
    observables::dynamics::correlation<velocity_autocorrelation<3, float> >::luaopen(L);
    observables::dynamics::correlation<velocity_autocorrelation<2, float> >::luaopen(L);
    return 0;
}

// explicit instantiation
template class velocity_autocorrelation<3, double>;
template class velocity_autocorrelation<2, double>;
template class velocity_autocorrelation<3, float>;
template class velocity_autocorrelation<2, float>;

} // namespace dynamics
} // namespace host

namespace dynamics
{

// explicit instantiation
template class correlation<host::dynamics::velocity_autocorrelation<3, double> >;
template class correlation<host::dynamics::velocity_autocorrelation<2, double> >;
template class correlation<host::dynamics::velocity_autocorrelation<3, float> >;
template class correlation<host::dynamics::velocity_autocorrelation<2, float> >;

} // namespace dynamics
} // namespace observables
} // namespace halmd
