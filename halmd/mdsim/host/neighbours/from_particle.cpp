/*
 * Copyright © 2008-2011  Peter Colberg
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

#include <halmd/mdsim/host/neighbours/from_particle.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace host {
namespace neighbours {

/**
 * construct neighbour list module
 *
 * @param particle mdsim::host::particle instance
 * @param box mdsim::box instance
 * @param cutoff force cutoff radius
 * @param skin neighbour list skin
 */
template <int dimension, typename float_type>
from_particle<dimension, float_type>::from_particle(
    boost::shared_ptr<particle_type const> particle1
  , boost::shared_ptr<particle_type const> particle2
  , boost::shared_ptr<box_type const> box
  , matrix_type const& r_cut
  , double skin
  , boost::shared_ptr<logger> logger
)
  // dependency injection
  : particle1_(particle1)
  , particle2_(particle2)
  , box_(box)
  , logger_(logger)
  // allocate parameters
  , neighbour_(particle1_->nparticle())
  , r_skin_(skin)
  , rr_cut_skin_(particle1_->nspecies(), particle2_->nspecies())
{
    matrix_type r_cut_skin(r_cut.size1(), r_cut.size2());
    typename matrix_type::value_type r_cut_max = 0;
    for (size_t i = 0; i < r_cut.size1(); ++i) {
        for (size_t j = 0; j < r_cut.size2(); ++j) {
            r_cut_skin(i, j) = r_cut(i, j) + r_skin_;
            rr_cut_skin_(i, j) = std::pow(r_cut_skin(i, j), 2);
            r_cut_max = max(r_cut_skin(i, j), r_cut_max);
        }
    }

    LOG("neighbour list skin: " << r_skin_);
}

/**
 * Update neighbour lists
 */
template <int dimension, typename float_type>
void from_particle<dimension, float_type>::update()
{
    on_prepend_update_();

    LOG_TRACE("update neighbour lists");

    scoped_timer_type timer(runtime_.update);

    typename particle_type::position_array_type const& position1 = particle1_->position();
    typename particle_type::position_array_type const& position2 = particle2_->position();
    typename particle_type::species_array_type const& species1 = particle1_->species();
    typename particle_type::species_array_type const& species2 = particle2_->species();

    // whether Newton's third law applies
    bool const reactio = (particle1_ == particle2_);

    for (size_t i = 0; i < particle1_->nparticle(); ++i) {
        // load first particle
        vector_type r1 = position1[i];
        unsigned int type1 = species1[i];

        // clear particle's neighbour list
        neighbour_[i].clear();

        for (size_t j = reactio ? (i + 1) : 0; j < particle2_->nparticle(); ++j) {
            // load second particle
            vector_type r2 = position2[j];
            unsigned int type2 = species2[j];

            // particle distance vector
            vector_type r = r1 - r2;
            box_->reduce_periodic(r);
            // squared particle distance
            float_type rr = inner_prod(r, r);

            // enforce cutoff radius with neighbour list skin
            if (rr >= rr_cut_skin_(type1, type2)) {
                continue;
            }

            // add particle to neighbour list
            neighbour_[i].push_back(j);
        }
    }

    on_append_update_();
}

template <typename neighbour_type>
static std::function<void ()>
wrap_update(boost::shared_ptr<neighbour_type> neighbour)
{
    return boost::bind(&neighbour_type::update, neighbour);
}

template <int dimension, typename float_type>
void from_particle<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    static string class_name("from_particle_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                namespace_("neighbours")
                [
                    class_<from_particle, boost::shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<
                            boost::shared_ptr<particle_type const>
                          , boost::shared_ptr<particle_type const>
                          , boost::shared_ptr<box_type const>
                          , matrix_type const&
                          , double
                          , boost::shared_ptr<logger_type>
                         >())
                        .property("r_skin", &from_particle::r_skin)
                        .property("update", &wrap_update<from_particle>)
                        .def("on_prepend_update", &from_particle::on_prepend_update)
                        .def("on_append_update", &from_particle::on_append_update)
                        .scope
                        [
                            class_<runtime>("runtime")
                                .def_readonly("update", &runtime::update)
                        ]
                        .def_readonly("runtime", &from_particle::runtime_)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_neighbours_from_particle(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    from_particle<3, double>::luaopen(L);
    from_particle<2, double>::luaopen(L);
#else
    from_particle<3, float>::luaopen(L);
    from_particle<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class from_particle<3, double>;
template class from_particle<2, double>;
#else
template class from_particle<3, float>;
template class from_particle<2, float>;
#endif

} // namespace neighbours
} // namespace host
} // namespace mdsim
} // namespace halmd
