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

#ifndef HALMD_MDSIM_HOST_NEIGHBOUR_HPP
#define HALMD_MDSIM_HOST_NEIGHBOUR_HPP

#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include <halmd/mdsim/force.hpp>
#include <halmd/mdsim/host/box.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/mdsim/neighbour.hpp>
#include <halmd/numeric/host/blas/vector.hpp>
#include <halmd/utility/options.hpp>

namespace halmd
{
namespace mdsim { namespace host
{

namespace sort
{
template <int dimension, typename float_type>
class hilbert;
}

template <int dimension, typename float_type>
class neighbour
  : public mdsim::neighbour<dimension>
{
public:
    // module definitions
    typedef neighbour _Self;
    typedef mdsim::neighbour<dimension> _Base;
    static void depends();
    static void select(po::options const& vm) {}
    static void options(po::options_description& desc);

    typedef host::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef mdsim::force<dimension> force_type;
    typedef host::box<dimension> box_type;
    typedef typename force_type::matrix_type matrix_type;

    typedef typename particle_type::neighbour_list cell_list;
    typedef boost::multi_array<cell_list, dimension> cell_lists;
    typedef numeric::host::blas::vector<size_t, dimension> cell_size_type;
    typedef numeric::host::blas::vector<ssize_t, dimension> cell_diff_type;

    shared_ptr<particle_type> particle;
    shared_ptr<force_type> force;
    shared_ptr<box_type> box;

    neighbour(modules::factory& factory, po::options const& vm);
    virtual ~neighbour() {}
    void update();
    bool check();

protected:
    friend class sort::hilbert<dimension, float_type>;

    void update_cells();
    void update_cell_neighbours(cell_size_type const& i);
    template <bool same_cell>
    void compute_cell_neighbours(size_t i, cell_list& c);

    /** neighbour list skin in MD units */
    float_type r_skin_;
    /** (cutoff lengths + neighbour list skin)² */
    matrix_type rr_cut_skin_;
    /** cell lists */
    cell_lists cell_;
    /** number of cells per dimension */
    cell_size_type ncell_;
    /** cell edge lengths */
    vector_type cell_length_;
    /** half neighbour list skin */
    float_type r_skin_half_;
    /** particle positions at last neighbour list update */
    std::vector<vector_type> r0_;
};

}} // namespace mdsim::host

} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_NEIGHBOUR_HPP */
