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

#ifndef HALMD_IO_TRAJECTORY_HDF5_WRITER_HPP
#define HALMD_IO_TRAJECTORY_HDF5_WRITER_HPP

#include <H5Cpp.h>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>

#include <halmd/io/trajectory/writer.hpp>
#include <halmd/mdsim/samples/host/trajectory.hpp>
#include <halmd/mdsim/particle.hpp>
#include <halmd/util/H5xx.hpp>
#include <halmd/utility/module.hpp>
#include <halmd/utility/options.hpp>

namespace halmd
{
namespace io { namespace trajectory { namespace writers
{

template <int dimension, typename float_type>
class hdf5
  : public trajectory::writer<dimension>
{
public:
    // module definitions
    typedef hdf5 _Self;
    typedef trajectory::writer<dimension> _Base;
    static void depends();

    typedef mdsim::samples::host::trajectory<dimension, float_type> sample_type;
    typedef typename sample_type::sample_vector sample_vector;
    typedef typename sample_type::sample_vector_ptr sample_vector_ptr;
    typedef boost::unordered_map<std::string, boost::function<void ()> > writer_map;

    /** returns file extension */
    std::string extension() const { return ".trj"; }

    hdf5(po::options const& vm);
    void append();
    void flush();

    shared_ptr<sample_type> sample;

private:
    H5::DataSet create_vector_dataset(H5::Group where, std::string const& name, sample_vector_ptr sample);
    H5::DataSet create_scalar_dataset(H5::Group where, std::string const& name, float_type sample);
    void write_vector_dataset(H5::DataSet dset, sample_vector_ptr sample);
    void write_scalar_dataset(H5::DataSet dset, double sample);

    /** absolute path to HDF5 trajectory file */
    boost::filesystem::path const path_;
    /** HDF5 file */
    H5::H5File file_;
    /** dataset write functors */
    writer_map writer_;
};

}}} // namespace io::trajectory::writers

} // namespace halmd

#endif /* ! HALMD_IO_TRAJECTORY_HDF5_WRITER_HPP */
