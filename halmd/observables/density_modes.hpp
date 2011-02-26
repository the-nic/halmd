/*
 * Copyright © 2011  Felix Höfling
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

#ifndef HALMD_OBSERVABLES_DENSITY_MODES_HPP
#define HALMD_OBSERVABLES_DENSITY_MODES_HPP

#include <lua.hpp>
#include <vector>

#include <halmd/observables/samples/density_modes.hpp>
#include <halmd/observables/utility/wavevectors.hpp>

namespace halmd
{
namespace observables
{

/**
 *  compute Fourier modes of the particle density
 *
 *  @f$ \rho_{\vec q} = \sum_{i=1}^N \exp(\textrm{i}\vec q \cdot \vec r_i) @f$
 *  for each particle species
 */
template <int dimension>
class density_modes
{
public:
    typedef observables::samples::density_modes<dimension> density_modes_sample_type;
    typedef typename density_modes_sample_type::mode_vector_vector_type result_type;
    typedef observables::utility::wavevectors<dimension> wavevectors_type;

    static void luaopen(lua_State* L);

    density_modes() {}
    virtual ~density_modes() {}
    virtual void acquire(double time) = 0;
    virtual result_type const& value() const = 0;
    virtual wavevectors_type const& wavevectors() const = 0;
    virtual std::vector<double> const& wavenumbers() const = 0;
};

} // namespace observables

} // namespace halmd

#endif /* ! HALMD_OBSERVABLES_DENSITY_MODES_HPP */
