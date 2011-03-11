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

#include <boost/foreach.hpp>

#include <halmd/io/logger.hpp>
#include <halmd/io/trajectory/writers/h5md.hpp>
#include <halmd/io/utility/hdf5.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>
#include <halmd/utility/demangle.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace io { namespace trajectory { namespace writers
{

/**
 * read sample from H5MD trajectory file
 */
template <int dimension, typename float_type>
h5md<dimension, float_type>::h5md(
    shared_ptr<sample_type> sample
  , string const& file_name
)
  : sample(sample)
  , file_(file_name, H5F_ACC_TRUNC)
{
    LOG("write trajectory to file: " << file_.getFileName());

    // store file version in parameter group
    array<unsigned char, 2> version = {{ 1, 0 }};
    h5xx::write_attribute(h5xx::open_group(file_, "param"), "file_version", version);

    // open or create trajectory group
    H5::Group root = h5xx::open_group(file_, "trajectory");

    for (size_t i = 0; i < sample->r.size(); ++i) {
        H5::Group type;
        if (sample->r.size() > 1) {
            type = root.createGroup(string(1, 'A' + i));
        }
        else {
            type = root;
        }
        size_t size = sample->r[i]->size();
        H5::DataSet r = h5xx::create_dataset<sample_vector_type>(type, "position", size);
        H5::DataSet v = h5xx::create_dataset<sample_vector_type>(type, "velocity", size);

        // particle positions
        writers_.push_back(h5xx::make_dataset_writer(r, &*sample->r[i]));
        // particle velocities
        writers_.push_back(h5xx::make_dataset_writer(v, &*sample->v[i]));
    }

    // simulation time in reduced units
    H5::DataSet t = h5xx::create_dataset<double>(root, "time");
    writers_.push_back(h5xx::make_dataset_writer(t, &sample->time));
}

/**
 * append samples to datasets
 */
template <int dimension, typename float_type>
void h5md<dimension, float_type>::append(double time)
{
    on_append_(time);

    if (sample->time != time) {
        throw logic_error("phase space sample was not updated");
    }

    BOOST_FOREACH (boost::function<void ()> const& writer, writers_) {
        writer();
    }
}

/**
 * flush H5MD file
 */
template <int dimension, typename float_type>
void h5md<dimension, float_type>::flush()
{
    file_.flush(H5F_SCOPE_GLOBAL);
}

template <int dimension, typename float_type>
void h5md<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("h5md_" + lexical_cast<string>(dimension) + "_" + demangled_name<float_type>() + "_");
    module(L)
    [
        namespace_("halmd_wrapper")
        [
            namespace_("io")
            [
                namespace_("trajectory")
                [
                    namespace_("writers")
                    [
                        class_<h5md, shared_ptr<_Base>, _Base>(class_name.c_str())
                            .def(constructor<
                                shared_ptr<sample_type>
                              , string const&
                            >())
                            .def("file", &h5md::file)
                    ]
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
        &h5md<3, double>::luaopen
    ]
    [
        &h5md<2, double>::luaopen
    ]
    [
        &h5md<3, float>::luaopen
    ]
    [
        &h5md<2, float>::luaopen
    ];
}

} // namespace

}}} // namespace io::trajectory::writers

} // namespace halmd
