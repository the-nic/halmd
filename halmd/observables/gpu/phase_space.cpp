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

#include <boost/make_shared.hpp>
#include <exception>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/particle_kernel.cuh>
#include <halmd/observables/gpu/phase_space.hpp>
#include <halmd/observables/gpu/phase_space_kernel.hpp>
#include <halmd/utility/demangle.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/scoped_timer.hpp>
#include <halmd/utility/timer.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace observables {
namespace gpu {

template <int dimension, typename float_type>
phase_space<gpu::samples::phase_space<dimension, float_type> >::phase_space(
    shared_ptr<sample_type> sample
  , shared_ptr<particle_type const> particle
  , shared_ptr<box_type const> box
  , shared_ptr<clock_type const> clock
  , shared_ptr<logger_type> logger
)
  : sample_(sample)
  , particle_(particle)
  , box_(box)
  , clock_(clock)
  , logger_(logger)
{
    try {
        cuda::copy(static_cast<vector_type>(box_->length()), phase_space_wrapper<dimension>::kernel.box_length);
    }
    catch (cuda::error const&)
    {
        LOG_ERROR("failed to copy box length to GPU");
        throw;
    }
}

template <int dimension, typename float_type>
phase_space<host::samples::phase_space<dimension, float_type> >::phase_space(
    shared_ptr<sample_type> sample
  , shared_ptr<particle_type /* FIXME const */> particle
  , shared_ptr<box_type const> box
  , shared_ptr<clock_type const> clock
  , shared_ptr<logger_type> logger
)
  : sample_(sample)
  , particle_(particle)
  , box_(box)
  , clock_(clock)
  , logger_(logger)
{
}

/**
 * Sample phase_space
 */
template <int dimension, typename float_type>
void phase_space<gpu::samples::phase_space<dimension, float_type> >::acquire()
{
    scoped_timer_type timer(runtime_.acquire);

    if (sample_->step == clock_->step()) {
        LOG_TRACE("sample is up to date");
        return;
    }

    LOG_TRACE("acquire GPU sample");

    // re-allocate memory which allows modules (e.g., dynamics::blocking_scheme)
    // to hold a previous copy of the sample
    {
        scoped_timer_type timer(runtime_.reset);
        sample_->reset();
    }

    phase_space_wrapper<dimension>::kernel.r.bind(particle_->g_r);
    phase_space_wrapper<dimension>::kernel.image.bind(particle_->g_image);
    phase_space_wrapper<dimension>::kernel.v.bind(particle_->g_v);

    cuda::configure(particle_->dim.grid, particle_->dim.block);
    phase_space_wrapper<dimension>::kernel.sample(
        particle_->g_reverse_tag
      , *sample_->r
      , *sample_->v
      , particle_->nbox
    );

    sample_->step = clock_->step();
}

/**
 * Sample phase_space
 */
template <int dimension, typename float_type>
void phase_space<host::samples::phase_space<dimension, float_type> >::acquire()
{
    scoped_timer_type timer(runtime_.acquire);

    if (sample_->step == clock_->step()) {
        LOG_TRACE("sample is up to date");
        return;
    }

    LOG_TRACE("acquire host sample");

    using mdsim::gpu::particle_kernel::untagged;

    try {
        cuda::copy(particle_->g_r, particle_->h_r);
        cuda::copy(particle_->g_image, particle_->h_image);
        cuda::copy(particle_->g_v, particle_->h_v);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to copy phase space from GPU to host");
        throw;
    }

    // re-allocate memory which allows modules (e.g., dynamics::blocking_scheme)
    // to hold a previous copy of the sample
    {
        scoped_timer_type timer(runtime_.reset);
        sample_->reset();
    }

    for (size_t i = 0; i < particle_->nbox; ++i) {
        unsigned int type, tag;
        vector_type r, v;
        tie(r, type) = untagged<vector_type>(particle_->h_r[i]);
        tie(v, tag) = untagged<vector_type>(particle_->h_v[i]);
        vector_type image = particle_->h_image[i];

        // periodically extended particle position
        assert(tag < sample_->r->size());
        box_->extend_periodic(r, image);
        (*sample_->r)[tag] = r;

        // particle velocity
        assert(tag < sample_->v->size());
        (*sample_->v)[tag] = v;

        // particle type
        assert(tag < sample_->type->size());
        (*sample_->type)[tag] = type;
    }
    sample_->step = clock_->step();
}

template <int dimension, typename float_type>
static int wrap_gpu_dimension(phase_space<gpu::samples::phase_space<dimension, float_type> > const&)
{
    return dimension;
}

template <int dimension, typename float_type>
void phase_space<gpu::samples::phase_space<dimension, float_type> >::luaopen(lua_State* L)
{
    using namespace luabind;
    static string const class_name(demangled_name<phase_space>());
    module(L, "libhalmd")
    [
        namespace_("observables")
        [
            class_<phase_space, shared_ptr<_Base>, _Base>(class_name.c_str())
                .property("dimension", &wrap_gpu_dimension<dimension, float_type>)
                .scope
                [
                    class_<runtime>("runtime")
                        .def_readonly("acquire", &runtime::acquire)
                        .def_readonly("reset", &runtime::reset)
                ]
                .def_readonly("runtime", &phase_space::runtime_)

          , def("phase_space", &make_shared<phase_space
               , shared_ptr<sample_type>
               , shared_ptr<particle_type const>
               , shared_ptr<box_type const>
               , shared_ptr<clock_type const>
               , shared_ptr<logger_type>
            >)
        ]
    ];
}

template <int dimension, typename float_type>
static int wrap_host_dimension(phase_space<host::samples::phase_space<dimension, float_type> > const&)
{
    return dimension;
}

template <int dimension, typename float_type>
void phase_space<host::samples::phase_space<dimension, float_type> >::luaopen(lua_State* L)
{
    using namespace luabind;
    static string const class_name(demangled_name<phase_space>());
    module(L, "libhalmd")
    [
        namespace_("observables")
        [
            class_<phase_space, shared_ptr<_Base>, _Base>(class_name.c_str())
                .property("dimension", &wrap_host_dimension<dimension, float_type>)
                .scope
                [
                    class_<runtime>("runtime")
                        .def_readonly("acquire", &runtime::acquire)
                        .def_readonly("reset", &runtime::reset)
                ]
                .def_readonly("runtime", &phase_space::runtime_)

          , def("phase_space", &make_shared<phase_space
               , shared_ptr<sample_type>
               , shared_ptr<particle_type /* FIXME const */>
               , shared_ptr<box_type const>
               , shared_ptr<clock_type const>
               , shared_ptr<logger_type>
            >)
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_observables_gpu_phase_space(lua_State* L)
{
    phase_space<gpu::samples::phase_space<3, float> >::luaopen(L);
    phase_space<gpu::samples::phase_space<2, float> >::luaopen(L);
    phase_space<host::samples::phase_space<3, float> >::luaopen(L);
    phase_space<host::samples::phase_space<2, float> >::luaopen(L);
    return 0;
}

// explicit instantiation
template class phase_space<gpu::samples::phase_space<3, float> >;
template class phase_space<gpu::samples::phase_space<2, float> >;
template class phase_space<host::samples::phase_space<3, float> >;
template class phase_space<host::samples::phase_space<2, float> >;

} // namespace observables
} // namespace gpu
} // namespace halmd
