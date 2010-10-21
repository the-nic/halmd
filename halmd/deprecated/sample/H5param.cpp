/* HDF5 parameter group
 *
 * Copyright © 2008-2009  Peter Colberg
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

#include <halmd/sample/H5param.hpp>

namespace halmd
{

/**
 * initialize HDF5 parameter group
 */
H5param::H5param(H5::H5File file)
{
    try {
        H5XX_NO_AUTO_PRINT(H5::FileIException);
        H5::Group::operator=(file.openGroup("param"));
    }
    catch (H5::FileIException const&) {
        H5::Group::operator=(file.createGroup("param"));
    }
}

/**
 * create or open HDF5 group
 */
H5xx::group H5param::operator[](std::string const& name)
{
    try {
        H5XX_NO_AUTO_PRINT(H5::GroupIException);
        return openGroup(name);
    }
    catch (H5::GroupIException const&) {
        return createGroup(name);
    }
}

} // namespace halmd