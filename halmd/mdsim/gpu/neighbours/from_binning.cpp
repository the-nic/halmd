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

#include <boost/bind.hpp>

#include <halmd/mdsim/gpu/neighbours/from_binning.hpp>
#include <halmd/mdsim/gpu/neighbours/from_binning_kernel.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/signal.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {
namespace gpu {
namespace neighbours {

/**
 * construct neighbour list module
 *
 * @param particle mdsim::gpu::particle instance
 * @param box mdsim::box instance
 * @param cutoff force cutoff radius
 * @param skin neighbour list skin
 * @param cell_occupancy desired average cell occupancy
 */
template <int dimension, typename float_type>
from_binning<dimension, float_type>::from_binning(
    boost::shared_ptr<particle_type const> particle1
  , boost::shared_ptr<particle_type const> particle2 /* FIXME not implemented */
  , boost::shared_ptr<binning_type const> binning1
  , boost::shared_ptr<binning_type const> binning2 /* FIXME not implemented */
  , boost::shared_ptr<box_type const> box
  , matrix_type const& r_cut
  , double skin
  , boost::shared_ptr<logger> logger
  , double cell_occupancy
)
  // dependency injection
  : particle_(particle1)
  , box_(box)
  , binning_(binning1)
  , logger_(logger)
  // allocate parameters
  , r_skin_(skin)
  , rr_cut_skin_(r_cut.size1(), r_cut.size2())
  , g_rr_cut_skin_(rr_cut_skin_.data().size())
  , nu_cell_(cell_occupancy) // FIXME neighbour list occupancy
{
    typename matrix_type::value_type r_cut_max = 0;
    for (size_t i = 0; i < r_cut.size1(); ++i) {
        for (size_t j = 0; j < r_cut.size2(); ++j) {
            rr_cut_skin_(i, j) = std::pow(r_cut(i, j) + r_skin_, 2);
            r_cut_max = max(r_cut(i, j), r_cut_max);
        }
    }

    // volume of n-dimensional sphere with neighbour list radius
    // volume of unit sphere: V_d = π^(d/2) / Γ(1+d/2), Γ(1) = 1, Γ(1/2) = √π
    float_type unit_sphere[5] = {0, 2, M_PI, 4 * M_PI / 3, M_PI * M_PI / 2 };
    assert(dimension <= 4);
    float_type neighbour_sphere = unit_sphere[dimension] * pow(r_cut_max + r_skin_, dimension);
    // partial number density
    float_type density = particle_->nparticle() / box_->volume();
    // number of placeholders per neighbour list
    size_ = static_cast<size_t>(ceil(neighbour_sphere * (density / binning_->effective_cell_occupancy())));
    // at least cell_size (or warp_size?) placeholders
    // FIXME what is a sensible lower bound?
    size_ = max(size_, binning_->cell_size());
    // number of neighbour lists
    stride_ = particle_->dim.threads();
    // allocate neighbour lists
    g_neighbour_.resize(stride_ * size_);

    LOG("neighbour list skin: " << r_skin_);
    LOG("number of placeholders per neighbour list: " << size_);

    try {
        cuda::copy(particle_->nparticle(), get_from_binning_kernel<dimension>().nbox);
        cuda::copy(rr_cut_skin_.data(), g_rr_cut_skin_);
        cuda::copy(size_, get_from_binning_kernel<dimension>().neighbour_size);
        cuda::copy(stride_, get_from_binning_kernel<dimension>().neighbour_stride);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to copy neighbour list parameters to device symbols");
        throw;
    }
}

/**
 * Update neighbour lists
 */
template <int dimension, typename float_type>
void from_binning<dimension, float_type>::update()
{
    // Emit on_prepend_update signal, which may be connected e.g. to the
    // binning update slot. We don't call binning::update directly, since
    // the order of calls is setup at the Lua level, and it allows us to
    // pass binning as a const dependency.
    on_prepend_update_();

    LOG_TRACE("update neighbour lists");

    scoped_timer_type timer(runtime_.update);

    // mark neighbour list placeholders as virtual particles
    cuda::memset(g_neighbour_, 0xFF);
    // build neighbour lists
    cuda::vector<int> g_ret(1);
    cuda::host::vector<int> h_ret(1);
    cuda::memset(g_ret, EXIT_SUCCESS);
    cuda::configure(binning_->dim_cell().grid, binning_->dim_cell().block, binning_->cell_size() * (2 + dimension) * sizeof(int));
    get_from_binning_kernel<dimension>().r.bind(particle_->position());
    get_from_binning_kernel<dimension>().rr_cut_skin.bind(g_rr_cut_skin_);
    get_from_binning_kernel<dimension>().update_neighbours(
        g_ret, g_neighbour_, binning_->g_cell()
      , rr_cut_skin_.size1(), rr_cut_skin_.size2()
      , binning_->ncell()
      , static_cast<vector_type>(box_->length())
    );
    cuda::thread::synchronize();
    cuda::copy(g_ret, h_ret);
    if (h_ret.front() != EXIT_SUCCESS) {
        throw std::runtime_error("overcrowded placeholders in neighbour lists update");
    }

    on_append_update_();
}

template <int dimension, typename float_type>
float_type from_binning<dimension, float_type>::defaults::occupancy() {
    return 0.4;
}

template <typename neighbour_type>
static boost::function<void ()>
wrap_update(boost::shared_ptr<neighbour_type> neighbour)
{
    return boost::bind(&neighbour_type::update, neighbour);
}

template <int dimension, typename float_type>
void from_binning<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("from_binning_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("neighbours")
                [
                    class_<from_binning, boost::shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<
                            boost::shared_ptr<particle_type const>
                          , boost::shared_ptr<particle_type const>
                          , boost::shared_ptr<binning_type const>
                          , boost::shared_ptr<binning_type const>
                          , boost::shared_ptr<box_type const>
                          , matrix_type const&
                          , double
                          , boost::shared_ptr<logger_type>
                          , double
                        >())
                        .property("r_skin", &from_binning::r_skin)
                        .property("cell_occupancy", &from_binning::cell_occupancy)
                        .property("update", &wrap_update<from_binning>)
                        .def("on_prepend_update", &from_binning::on_prepend_update)
                        .def("on_append_update", &from_binning::on_append_update)
                        .scope
                        [
                            namespace_("defaults")
                            [
                                def("occupancy", &defaults::occupancy)
                            ]

                          , class_<runtime>("runtime")
                                .def_readonly("update", &runtime::update)
                        ]
                        .def_readonly("runtime", &from_binning::runtime_)
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_neighbours_from_binning(lua_State* L)
{
    from_binning<3, float>::luaopen(L);
    from_binning<2, float>::luaopen(L);
    return 0;
}

// explicit instantiation
template class from_binning<3, float>;
template class from_binning<2, float>;

} // namespace neighbours
} // namespace gpu
} // namespace mdsim
} // namespace halmd
