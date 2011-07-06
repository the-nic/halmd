/*
 * Copyright © 2011  Peter Colberg
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

#define BOOST_TEST_MODULE h5md
#include <boost/test/unit_test.hpp>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <ctime>

#include <halmd/io/writers/h5md.hpp>

using namespace boost;
using namespace halmd;
using namespace halmd::io; // avoid ambiguity of io:: between halmd::io and boost::io
using namespace std;

struct writer
{
    typedef writers::h5md::version_type version_type;
    /** create H5MD file */
    writer() {
        file = make_shared<writers::h5md>("h5md.h5");
    }
    /** close and unlink H5MD file */
    ~writer() {
        file.reset();
        filesystem::remove("h5md.h5");
    }
    shared_ptr<writers::h5md> file;
};

BOOST_FIXTURE_TEST_CASE( check_version, writer )
{
    H5::Group attr = file->file().openGroup("h5md");
    version_type version = h5xx::read_attribute<version_type>(attr, "version");
    BOOST_CHECK_EQUAL( version[0], file->version()[0] );
    BOOST_CHECK_EQUAL( version[1], file->version()[1] );
}

BOOST_FIXTURE_TEST_CASE( read_attributes, writer )
{
    H5::Group attr = file->file().openGroup("h5md");
    version_type version = h5xx::read_attribute<version_type>(attr, "version");
    BOOST_TEST_MESSAGE( "H5MD major version:\t" << version[0] );
    BOOST_TEST_MESSAGE( "H5MD minor version:\t" << version[1] );
    time_t creation_time = h5xx::read_attribute<time_t>(attr, "creation_time");
    char creation_time_fmt[256];
    BOOST_CHECK( strftime(creation_time_fmt, sizeof(creation_time_fmt), "%c", localtime(&creation_time)) );
    BOOST_TEST_MESSAGE( "H5MD creation time:\t" << creation_time_fmt );
    BOOST_TEST_MESSAGE( "H5MD creator:\t\t" << h5xx::read_attribute<string>(attr, "creator") );
    BOOST_TEST_MESSAGE( "H5MD creator version:\t" << h5xx::read_attribute<string>(attr, "creator_version") );
}

BOOST_FIXTURE_TEST_CASE( flush_file, writer )
{
    BOOST_CHECK_NO_THROW( file->flush() );
}

BOOST_FIXTURE_TEST_CASE( close_file, writer )
{
    BOOST_CHECK_NO_THROW( file->close() );
}
