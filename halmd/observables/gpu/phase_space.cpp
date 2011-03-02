/*
 * Copyright © 2008-2010  Peter Colberg and Felix Höfling
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

#include <exception>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/particle_kernel.cuh>
#include <halmd/observables/gpu/phase_space.hpp>
#include <halmd/observables/gpu/phase_space_kernel.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace observables { namespace gpu
{

template <int dimension, typename float_type>
phase_space<gpu::samples::phase_space<dimension, float_type> >::phase_space(
    shared_ptr<sample_type> sample
  , shared_ptr<particle_type> particle
  , shared_ptr<box_type> box
)
  : sample(sample)
  , particle(particle)
  , box(box)
{
}

template <int dimension, typename float_type>
phase_space<host::samples::phase_space<dimension, float_type> >::phase_space(
    shared_ptr<sample_type> sample
  , shared_ptr<particle_type> particle
  , shared_ptr<box_type> box
)
  : sample(sample)
  , particle(particle)
  , box(box)
{
}


/**
 * Sample phase_space
 */
template <int dimension, typename float_type>
void phase_space<gpu::samples::phase_space<dimension, float_type> >::acquire(double time)
{
    LOG_TRACE("[phase_space] acquire GPU sample");
    // FIXME

    sample->time = time;
}

/**
 * Sample phase_space
 */
template <int dimension, typename float_type>
void phase_space<host::samples::phase_space<dimension, float_type> >::acquire(double time)
{
    LOG_TRACE("[phase_space] acquire host sample");

    using mdsim::gpu::particle_kernel::untagged;

    try {
        cuda::copy(particle->g_r, particle->h_r);
        cuda::copy(particle->g_image, particle->h_image);
        cuda::copy(particle->g_v, particle->h_v);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw runtime_error("failed to copy phase space from GPU to host");
    }

    for (size_t i = 0; i < particle->nbox; ++i) {
        unsigned int type, tag;
        vector_type r, v;
        tie(r, type) = untagged<vector_type>(particle->h_r[i]);
        tie(v, tag) = untagged<vector_type>(particle->h_v[i]);
        vector_type image = particle->h_image[i];
        vector_type L = static_cast<vector_type>(box->length());
        // periodically extended particle position
        (*sample->r[type])[tag] = r + element_prod(image, L);
        // particle velocity
        (*sample->v[type])[tag] = v;
    }
    sample->time = time;
}

template <int dimension, typename float_type>
void phase_space<gpu::samples::phase_space<dimension, float_type> >::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("phase_space_" + lexical_cast<string>(dimension) + "_");
    module(L, "halmd_wrapper")
    [
        namespace_("observables")
        [
            namespace_("gpu")
            [
                namespace_("gpu")
                [
                    class_<phase_space, shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<
                             shared_ptr<sample_type>
                           , shared_ptr<particle_type>
                           , shared_ptr<box_type>
                        >())
                ]
            ]
        ]
    ];
}

template <int dimension, typename float_type>
void phase_space<host::samples::phase_space<dimension, float_type> >::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("phase_space_" + lexical_cast<string>(dimension) + "_");
    module(L, "halmd_wrapper")
    [
        namespace_("observables")
        [
            namespace_("gpu")
            [
                namespace_("host")
                [
                    class_<phase_space, shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<
                             shared_ptr<sample_type>
                           , shared_ptr<particle_type>
                           , shared_ptr<box_type>
                        >())
                ]
            ]
        ]
    ];
}

namespace // limit symbols to translation unit
{

__attribute__((constructor)) void register_lua()
{
    lua_wrapper::register_(1) //< distance of derived to base class
    [
        &phase_space<gpu::samples::phase_space<3, float> >::luaopen
    ]
    [
        &phase_space<gpu::samples::phase_space<2, float> >::luaopen
    ]
    [
        &phase_space<host::samples::phase_space<3, float> >::luaopen
    ]
    [
        &phase_space<host::samples::phase_space<2, float> >::luaopen
    ];
}

} // namespace

// explicit instantiation
template class phase_space<gpu::samples::phase_space<3, float> >;
template class phase_space<gpu::samples::phase_space<2, float> >;
template class phase_space<host::samples::phase_space<3, float> >;
template class phase_space<host::samples::phase_space<2, float> >;

}} // namespace observables::gpu

} // namespace halmd