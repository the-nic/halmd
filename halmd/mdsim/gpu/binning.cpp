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

#include <exception>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/gpu/binning.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/scoped_timer.hpp>
#include <halmd/utility/timer.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace mdsim { namespace gpu
{

/**
 * construct particle binning module
 *
 * @param particle mdsim::gpu::particle instance
 * @param box mdsim::box instance
 * @param cutoff force cutoff radius
 * @param skin neighbour list skin
 * @param cell_occupancy desired average cell occupancy
 */
template <int dimension, typename float_type>
binning<dimension, float_type>::binning(
    shared_ptr<particle_type> particle
  , shared_ptr<box_type> box
  , matrix_type const& r_cut
  , double skin
  , double cell_occupancy
)
  // dependency injection
  : particle(particle)
  , box(box)
  // allocate parameters
  , r_skin_(skin)
  , rr_skin_half_(pow(r_skin_ / 2, 2))
  , rr_cut_skin_(particle->ntype, particle->ntype)
  , nu_cell_(cell_occupancy)
  , sort_(particle->nbox, particle->dim.threads_per_block())
{
    typename matrix_type::value_type r_cut_max = 0;
    for (size_t i = 0; i < particle->ntype; ++i) {
        for (size_t j = i; j < particle->ntype; ++j) {
            rr_cut_skin_(i, j) = std::pow(r_cut(i, j) + r_skin_, 2);
            r_cut_max = max(r_cut(i, j), r_cut_max);
        }
    }
    // find an optimal(?) cell size
    // ideally, we would like to have warp_size placeholders per cell
    //
    // definitions:
    // a) n_cell = L_box / cell_length   (for each dimension)
    // b) #cells = prod(n_cell)
    // c) #particles = #cells × cell_size × ν_eff
    //
    // constraints:
    // 1) cell_length > r_c + r_skin  (potential cutoff + neighbour list skin)
    // 2) cell_size is a (small) multiple of warp_size
    // 3) n_cell respects the aspect ratios of the simulation box
    // 4) ν_eff ≤ ν  (actual vs. desired occupancy)
    //
    // upper bound on ncell_ provided by 1)
    cell_size_type ncell_max =
        static_cast<cell_size_type>(box->length() / (r_cut_max + r_skin_));
    // determine optimal value from 2,3) together with b,c)
    size_t warp_size = cuda::device::properties(cuda::device::get()).warp_size();
    double nwarps = particle->nbox / (nu_cell_ * warp_size);
    double volume = accumulate(box->length().begin(), box->length().end(), 1., multiplies<double>());
    ncell_ = static_cast<cell_size_type>(ceil(box->length() * pow(nwarps / volume, 1./dimension)));
    LOG_DEBUG("desired values for number of cells: " << ncell_);
    LOG_DEBUG("upper bound on number of cells: " << ncell_max);
    // respect upper bound
    ncell_ = element_min(ncell_, ncell_max);

    // compute derived values
    size_t ncells = accumulate(ncell_.begin(), ncell_.end(), 1, multiplies<size_t>());
    cell_size_ = warp_size * static_cast<size_t>(ceil(nwarps / ncells));
    vector_type cell_length_ =
        element_div(static_cast<vector_type>(box->length()), static_cast<vector_type>(ncell_));
    dim_cell_ = cuda::config(
        dim3(
             accumulate(ncell_.begin(), ncell_.end() - 1, 1, multiplies<size_t>())
           , ncell_.back()
        )
      , cell_size_
    );

    if (*min_element(ncell_.begin(), ncell_.end()) < 3) {
        throw std::logic_error("number of cells per dimension must be at least 3");
    }

    LOG("number of placeholders per cell: " << cell_size_);
    LOG("number of cells per dimension: " << ncell_);
    LOG("cell edge lengths: " << cell_length_);
    LOG("desired average cell occupancy: " << nu_cell_);
    double nu_cell_eff = static_cast<double>(particle->nbox) / dim_cell_.threads();
    LOG("effective average cell occupancy: " << nu_cell_eff);

    try {
        cuda::copy(particle->nbox, get_binning_kernel<dimension>().nbox);
        cuda::copy(static_cast<fixed_vector<uint, dimension> >(ncell_), get_binning_kernel<dimension>().ncell);
        cuda::copy(cell_length_, get_binning_kernel<dimension>().cell_length);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to copy cell parameters to device symbols");
        throw;
    }

    try {
        g_cell_.resize(dim_cell_.threads());
        g_cell_offset_.resize(dim_cell_.blocks_per_grid());
        g_cell_index_.reserve(particle->dim.threads());
        g_cell_index_.resize(particle->nbox);
        g_cell_permutation_.reserve(particle->dim.threads());
        g_cell_permutation_.resize(particle->nbox);
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to allocate cell placeholders in global device memory");
        throw;
    }
}

/**
 * register module runtime accumulators
 */
template <int dimension, typename float_type>
void binning<dimension, float_type>::register_runtimes(profiler_type& profiler)
{
    profiler.register_runtime(runtime_.update, "update", "cell lists update");
}

/**
 * Update cell lists
 */
template <int dimension, typename float_type>
void binning<dimension, float_type>::update()
{
    scoped_timer<timer> timer_(runtime_.update);

    // compute cell indices for particle positions
    cuda::configure(particle->dim.grid, particle->dim.block);
    get_binning_kernel<dimension>().compute_cell(particle->g_r, g_cell_index_);

    // generate permutation
    cuda::configure(particle->dim.grid, particle->dim.block);
    get_binning_kernel<dimension>().gen_index(g_cell_permutation_);
    sort_(g_cell_index_, g_cell_permutation_);

    // compute global cell offsets in sorted particle list
    cuda::memset(g_cell_offset_, 0xFF);
    cuda::configure(particle->dim.grid, particle->dim.block);
    get_binning_kernel<dimension>().find_cell_offset(g_cell_index_, g_cell_offset_);

    // assign particles to cells
    cuda::vector<int> g_ret(1);
    cuda::host::vector<int> h_ret(1);
    cuda::memset(g_ret, EXIT_SUCCESS);
    cuda::configure(dim_cell_.grid, dim_cell_.block);
    get_binning_kernel<dimension>().assign_cells(g_ret, g_cell_index_, g_cell_offset_, g_cell_permutation_, g_cell_);
    cuda::copy(g_ret, h_ret);
    if (h_ret.front() != EXIT_SUCCESS) {
        throw std::runtime_error("overcrowded placeholders in cell lists update");
    }
}

template <int dimension, typename float_type>
void binning<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("binning_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                class_<binning, shared_ptr<binning> >(class_name.c_str())
                    .def(constructor<
                        shared_ptr<particle_type>
                      , shared_ptr<box_type>
                      , matrix_type const&
                      , double
                      , double
                    >())
                    .def("register_runtimes", &binning::register_runtimes)
                    .property("r_skin", &binning::r_skin)
                    .property("cell_occupancy", &binning::cell_occupancy)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_binning(lua_State* L)
{
    binning<3, float>::luaopen(L);
    binning<2, float>::luaopen(L);
    return 0;
}

// explicit instantiation
template class binning<3, float>;
template class binning<2, float>;

}} // namespace mdsim::gpu

} // namespace halmd
