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

#ifndef HALMD_MDSIM_HOST_BINNING_HPP
#define HALMD_MDSIM_HOST_BINNING_HPP

#include <boost/make_shared.hpp>
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/shared_ptr.hpp>
#include <lua.hpp>
#include <vector>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace mdsim {
namespace host {

template <int dimension, typename float_type>
class binning
{
public:
    typedef host::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef boost::numeric::ublas::symmetric_matrix<float_type, boost::numeric::ublas::lower> matrix_type;
    typedef mdsim::box<dimension> box_type;
    typedef logger logger_type;

    typedef std::vector<unsigned int> cell_list;
    typedef boost::multi_array<cell_list, dimension> cell_lists;
    typedef fixed_vector<size_t, dimension> cell_size_type;
    typedef fixed_vector<ssize_t, dimension> cell_diff_type;

    static void luaopen(lua_State* L);

    binning(
        boost::shared_ptr<particle_type const> particle
      , boost::shared_ptr<box_type const> box
      , matrix_type const& r_cut
      , float_type skin
      , boost::shared_ptr<logger_type> logger = boost::make_shared<logger_type>()
    );
    virtual void update();

    //! returns neighbour list skin in MD units
    float_type r_skin() const
    {
        return r_skin_;
    }

    //! cell edge length
    vector_type const& cell_length() const
    {
        return cell_length_;
    }

    //! number of cells per dimension
    cell_size_type ncell() const
    {
        return ncell_;
    }

    //! get cell lists
    cell_lists const& cell() const
    {
        return cell_;
    }

private:
    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;
    typedef typename profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type update;
    };

    //! system state
    boost::shared_ptr<particle_type const> particle_;
    /** module logger */
    boost::shared_ptr<logger_type> logger_;
    /** neighbour list skin in MD units */
    float_type r_skin_;
    /** cell lists */
    cell_lists cell_;
    /** number of cells per dimension */
    cell_size_type ncell_;
    /** cell edge lengths */
    vector_type cell_length_;
    /** profiling runtime accumulators */
    runtime runtime_;
};

} // namespace host
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_BINNING_HPP */