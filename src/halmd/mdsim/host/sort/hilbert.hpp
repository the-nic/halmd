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

#ifndef HALMD_MDSIM_HOST_SORT_HILBERT_HPP
#define HALMD_MDSIM_HOST_SORT_HILBERT_HPP

#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/host/neighbour.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/mdsim/sort.hpp>
#include <halmd/utility/options.hpp>

namespace halmd
{
namespace mdsim { namespace host { namespace sort
{

template <int dimension, typename float_type>
class hilbert
  : public mdsim::sort<dimension>
{
public:
    // module definitions
    typedef hilbert _Self;
    typedef mdsim::sort<dimension> _Base;
    static void depends();
    static void options(po::options_description& desc) {}
    static void select(po::options const& vm) {}

    typedef host::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef mdsim::box<dimension> box_type;
    typedef host::neighbour<dimension, float_type> neighbour_type;

    typedef typename neighbour_type::cell_list cell_list;
    typedef typename neighbour_type::cell_size_type cell_size_type;

    shared_ptr<particle_type> particle;
    shared_ptr<box_type> box;
    shared_ptr<neighbour_type> neighbour;

    hilbert(modules::factory& factory, po::options const& vm);
    virtual ~hilbert() {}
    virtual void order();

protected:
    unsigned int map(vector_type r, unsigned int depth);
    void swap(unsigned int& v, unsigned int& a, unsigned int& b, unsigned int mask);

    /** 1-dimensional Hilbert curve mapping of cell lists */
    std::vector<cell_list*> cell_;
};

}}} // namespace mdsim::host::sort

} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_SORT_HILBERT_HPP */
