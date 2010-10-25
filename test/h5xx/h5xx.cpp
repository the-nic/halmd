/*
 * Copyright © 2010  Felix Höfling
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

#define BOOST_TEST_MODULE test_h5xx
#include <boost/test/unit_test.hpp>

#include <unistd.h>
#include <h5xx/h5xx.hpp>

#include <halmd/numeric/blas/fixed_vector.hpp>

BOOST_AUTO_TEST_CASE( h5xx_attribute )
{
    using namespace H5;
    using namespace h5xx;

    char const filename[] = "test_h5xx.hdf5";
    H5File file(filename, H5F_ACC_TRUNC);
    Group group = open_group(file, "/");

    uint64_t uint_value = 9223372036854775783;  // largest prime below 2^63
    h5xx::write_attribute(group, "integral, scalar", 1);   // store something of wrong type first
    h5xx::write_attribute(group, "integral, scalar", uint_value);  // overwrite value

    // long double is supported by the HDF5 library,
    // but neither by h5dump nor pytables ...
    long double ldouble_value = sqrtl(2.);
    h5xx::write_attribute(group, "long double, scalar", 2);   // store something of wrong type first
    h5xx::write_attribute(group, "long double, scalar", ldouble_value);
    h5xx::write_attribute(group, "double, scalar", static_cast<double>(ldouble_value));

    boost::array<char const*, 3> cstring_array = {{
        "HALMD", "HAL's MD package",
        "Highly accelerated large-scale molecular dynamics simulation package"
    }};
    typedef boost::array<std::string, 3> string_array_type;
    h5xx::write_attribute(group, "char [], scalar", cstring_array[1]);
    h5xx::write_attribute(group, "string, scalar", std::string(cstring_array[1]));
    h5xx::write_attribute(group, "char [], array", cstring_array);

    // use derived class instead of boost::array
    typedef halmd::fixed_vector<double, 5> double_array_type;
    double_array_type value_array;
    double double_values[] = {
       1., sqrt(2.), 2., sqrt(3.), 3.
    };
    std::copy(double_values, double_values + value_array.size(), value_array.data());
    h5xx::write_attribute(group, "double, array", value_array);

    typedef std::vector<double> double_vector_type;
    double_vector_type value_vector(value_array.size());
    std::copy(value_array.begin(), value_array.end(), value_vector.begin());
    h5xx::write_attribute(group, "double, std::vector", value_vector);
    value_vector.resize(4);
    h5xx::write_attribute(group, "double, std::vector", value_vector);     // overwrite with different size

    typedef boost::multi_array<int, 3> multi_array3;
    int data3[] = {
        99,98,97,96,
        95,94,93,92,
        91,90,89,88,

        87,86,85,84,
        83,82,81,80,
        79,78,77,76
    };
    multi_array3 multi_array_value(boost::extents[2][3][4]);
    multi_array_value.assign(data3, data3 + 2 * 3 * 4);
    h5xx::write_attribute(group, "int, multi_array", multi_array_value);

    // re-open file
    file.close();
    file.openFile(filename, H5F_ACC_RDONLY);
    group = open_group(file, "/");

    // check has_type<>
    BOOST_CHECK(has_type<uint64_t>(group.openAttribute("integral, scalar")));
    BOOST_CHECK(has_type<long double>(group.openAttribute("long double, scalar")));
    BOOST_CHECK(has_type<double>(group.openAttribute("double, scalar")));
    BOOST_CHECK(has_type<char const*>(group.openAttribute("char [], scalar")));
    BOOST_CHECK(has_type<std::string>(group.openAttribute("string, scalar")));
    BOOST_CHECK(has_type<string_array_type>(group.openAttribute("char [], array")));
    BOOST_CHECK(has_type<double_array_type>(group.openAttribute("double, array")));
    BOOST_CHECK(has_type<double_vector_type>(group.openAttribute("double, std::vector")));
    BOOST_CHECK(has_type<multi_array3>(group.openAttribute("int, multi_array")));

    // check has_scalar_space()
    BOOST_CHECK(has_scalar_space(group.openAttribute("integral, scalar")));
    BOOST_CHECK(has_scalar_space(group.openAttribute("long double, scalar")));
    BOOST_CHECK(has_scalar_space(group.openAttribute("double, scalar")));
    BOOST_CHECK(has_scalar_space(group.openAttribute("char [], scalar")));
    BOOST_CHECK(has_scalar_space(group.openAttribute("string, scalar")));

    // check has_extent<>
    BOOST_CHECK(has_extent<string_array_type>(group.openAttribute("char [], array")));
    BOOST_CHECK(has_extent<double_array_type>(group.openAttribute("double, array")));
    BOOST_CHECK(has_extent<multi_array3>(group.openAttribute("int, multi_array"),
                multi_array_value.shape()));

    // read attributes
    BOOST_CHECK(h5xx::read_attribute<uint64_t>(group, "integral, scalar") == uint_value);
    BOOST_CHECK(h5xx::read_attribute<long double>(group, "long double, scalar") == ldouble_value);
    BOOST_CHECK(h5xx::read_attribute<double>(group, "double, scalar")
                == static_cast<double>(ldouble_value));
    BOOST_CHECK(h5xx::read_attribute<std::string>(group, "char [], scalar") == cstring_array[1]);
    BOOST_CHECK(h5xx::read_attribute<std::string>(group, "string, scalar") == cstring_array[1]);
    // read support for fixed-size string array is missing
//     BOOST_CHECK(h5xx::read_attribute<const char*>(group, "char [], array") == cstring_array);
    BOOST_CHECK(h5xx::read_attribute<double_array_type>(group, "double, array") == value_array);
    value_vector = h5xx::read_attribute<double_vector_type>(group, "double, array");
    BOOST_CHECK(equal(value_vector.begin(), value_vector.end(), value_array.begin()));
    BOOST_CHECK(h5xx::read_attribute<multi_array3>(group, "int, multi_array") == multi_array_value);
    // read multi_array as linear vector
    std::vector<int> int_vector = h5xx::read_attribute<std::vector<int> >(group, "int, multi_array");
    BOOST_CHECK(int_vector.size() == multi_array_value.num_elements());
    BOOST_CHECK(equal(int_vector.begin(), int_vector.end(), multi_array_value.data()));

    // remove file
#ifndef NDEBUG
    unlink(filename);
#endif
}

// BOOST_CHECK doesn't like more than one template parameter :-(
// so we define these wrappers here
template <typename T>
inline bool has_extent_one_extra(H5::DataSet const& dataset)
{
    return h5xx::has_extent<T, 1>(dataset);
}

template <typename T>
inline bool has_extent_one_extra(H5::DataSet const& dataset, size_t const* shape)
{
    return h5xx::has_extent<T, 1>(dataset, shape);
}

BOOST_AUTO_TEST_CASE( h5xx_dataset )
{
    using namespace H5;
    using namespace h5xx;

    char const filename[] = "test_h5xx.hdf5";
    H5File file(filename, H5F_ACC_TRUNC);
    Group group = open_group(file, "/");

    //
    // create and write datasets
    //

    // scalar type
    uint64_t uint_value = 9223372036854775783;  // largest prime below 2^63
    make_dataset_writer(group, "uint", &uint_value)();
    // overwrite data set
    DataSet uint_dataset = create_dataset<uint64_t>(group, "uint");
    write(uint_dataset, uint_value);
    write(uint_dataset, uint_value + 1);

    // array type
    typedef boost::array<double, 3> array_type;
    array_type array_value = {{ 1, sqrt(2), 2 }};
    array_type array_value2 = {{ -1, sqrt(3), -3 }};
    DataSet array_dataset
        = create_dataset<array_type>(group, "array", 2);  // fixed size
    write(array_dataset, array_value, 0);           // write entry #0
    write(array_dataset, array_value2, 1);          // write entry #1

    // multi-array type
    typedef boost::multi_array<int, 2> multi_array2;
    int data2[] = {
        99,98,97,96,
        95,94,93,92,
        91,90,89,88,
    };
    multi_array2 multi_array_value(boost::extents[3][4]);
    multi_array_value.assign(data2, data2 + 3 * 4);
    DataSet multi_array_dataset
        = create_dataset<multi_array2>(group, "multi_array", multi_array_value.shape());
    write(multi_array_dataset, multi_array_value);    // append
    multi_array_value[1][2] = 1;
    write(multi_array_dataset, multi_array_value);    // append
    multi_array_value[1][2] = 2;
    make_dataset_write_at(multi_array_dataset, &multi_array_value)(0);  // overwrite first entry

    // vector of scalars
    std::vector<int> int_vector_value(data2, data2 + 3 * 4);
    DataSet int_vector_dataset
            = create_dataset<std::vector<int> >(group, "int_vector", int_vector_value.size());
    make_dataset_writer(int_vector_dataset, &int_vector_value)();

    // vector of arrays
    std::vector<array_type> array_vector_value;
    array_vector_value.push_back(array_value);
    array_vector_value.push_back(array_value2);
    DataSet array_vector_dataset
            = create_dataset<std::vector<array_type> >(group, "array_vector", array_vector_value.size());
    write(array_vector_dataset, array_vector_value);
    // write vector of wrong size
    array_vector_value.push_back(array_value2);
    BOOST_CHECK_THROW(write(array_vector_dataset, array_vector_value), std::runtime_error);
    array_vector_value.pop_back();

    // write to dataset of wrong type and size
    BOOST_CHECK_THROW(write(int_vector_dataset, array_vector_value), std::runtime_error);

    // re-open file
    file.flush(H5F_SCOPE_GLOBAL);
    file.close();
    file.openFile(filename, H5F_ACC_RDONLY);
    group = open_group(file, "/");

    //
    // read datasets
    //

    // scalar type dataset
    uint_dataset = group.openDataSet("uint");
    BOOST_CHECK(has_type<uint64_t>(uint_dataset));
    BOOST_CHECK(elements(uint_dataset) == 2);

    uint64_t uint_value_;
    read(uint_dataset, &uint_value_, 0);
    BOOST_CHECK(uint_value_ == uint_value);
    read(uint_dataset, &uint_value_, 1);
    BOOST_CHECK(uint_value_ == uint_value + 1);
    read(uint_dataset, &uint_value_, -1);
    BOOST_CHECK(uint_value_ == uint_value + 1);
    read(uint_dataset, &uint_value_, -2);
    BOOST_CHECK(uint_value_ == uint_value);
    BOOST_CHECK_THROW(read(uint_dataset, &uint_value_, 2), std::runtime_error);
    BOOST_CHECK_THROW(read(uint_dataset, &uint_value_, -3), std::runtime_error);

    // array type dataset
    array_dataset = group.openDataSet("array");
    BOOST_CHECK(has_type<array_type>(array_dataset));
    BOOST_CHECK(has_extent_one_extra<array_type>(array_dataset));
    BOOST_CHECK(elements(array_dataset) == 2 * 3);
    array_type array_value_;
    read(array_dataset, &array_value_, 0);
    BOOST_CHECK(array_value_ == array_value);
    read(array_dataset, &array_value_, 1);
    BOOST_CHECK(array_value_ == array_value2);

    // multi-array type dataset
    multi_array_dataset = group.openDataSet("multi_array");
    BOOST_CHECK(has_type<multi_array2>(multi_array_dataset));
    BOOST_CHECK(has_extent_one_extra<multi_array2>(multi_array_dataset, multi_array_value.shape()));
    BOOST_CHECK(elements(multi_array_dataset) == 2 * 3 * 4);
    multi_array2 multi_array_value_;
    read(multi_array_dataset, &multi_array_value_, 0);
    multi_array_value[1][2] = 2;
    BOOST_CHECK(multi_array_value_ == multi_array_value);
    read(multi_array_dataset, &multi_array_value_, 1);
    multi_array_value[1][2] = 1;
    BOOST_CHECK(multi_array_value_ == multi_array_value);

    // vector of scalars
    int_vector_dataset = group.openDataSet("int_vector");
    std::vector<int> int_vector_value_;
    read(int_vector_dataset, &int_vector_value_, 0);
    BOOST_CHECK(int_vector_value_.size() == int_vector_value.size());
    BOOST_CHECK(std::equal(
        int_vector_value_.begin()
      , int_vector_value_.end()
      , int_vector_value.begin()
    ));

    // vector of arrays
    array_vector_dataset = group.openDataSet("array_vector");
    std::vector<array_type> array_vector_value_;
    read(array_vector_dataset, &array_vector_value_, 0);
    BOOST_CHECK(array_vector_value_.size() == array_vector_value.size());
    BOOST_CHECK(std::equal(
        array_vector_value_.begin()
      , array_vector_value_.end()
      , array_vector_value.begin()
    ));

    // remove file
#ifndef NDEBUG
    unlink(filename);
#endif
}

BOOST_AUTO_TEST_CASE( h5xx_group )
{
    using namespace H5;
    using namespace h5xx;

    char const filename[] = "test_h5xx.hdf5";
    H5File file(filename, H5F_ACC_TRUNC);

    BOOST_CHECK_NO_THROW(open_group(file, "/"));
    BOOST_CHECK_NO_THROW(open_group(file, ""));

    Group group = open_group(file, "/");
    BOOST_CHECK_NO_THROW(open_group(group, "/"));
    BOOST_CHECK_NO_THROW(open_group(group, ""));

    // create a hierarchy with attributes
    h5xx::write_attribute(open_group(file, "/"), "level", 0);
    h5xx::write_attribute(open_group(file, "/one"), "level", 1);
    h5xx::write_attribute(open_group(file, "/one/two"), "level", 2);
    h5xx::write_attribute(open_group(file, "/one/two/three"), "level", 3);

    group = open_group(file, "one");
    BOOST_CHECK(group.getNumAttrs() == 1);
    BOOST_CHECK(h5xx::read_attribute<int>(group, "level") == 1);

    open_group(group, "branch");        // create branch in '/one'
    BOOST_CHECK(group.getNumAttrs() == 1);
    BOOST_CHECK(group.getNumObjs() == 2);

    group = open_group(group, "two/three/");
    BOOST_CHECK(group.getNumAttrs() == 1);
    BOOST_CHECK(h5xx::read_attribute<int>(group, "level") == 3);

    group = open_group(file, "one/two");
    BOOST_CHECK(group.getNumAttrs() == 1);
    BOOST_CHECK(h5xx::read_attribute<int>(group, "level") == 2);

    group = open_group(group, "three");
    BOOST_CHECK(group.getNumAttrs() == 1);
    BOOST_CHECK(h5xx::read_attribute<int>(group, "level") == 3);

    group = open_group(group, "three");          // create new group
    BOOST_CHECK(group.getNumAttrs() == 0);
    BOOST_CHECK_THROW(h5xx::read_attribute<int>(group, "level"), AttributeIException);

    // test H5::path
    BOOST_CHECK(path(open_group(file, "/")) == "/");
    BOOST_CHECK(path(open_group(file, "one/two/three")) == "/one/two/three");
    // note on previous check: semantics of open_group seems to be somewhat too lazy

    file.close();

    // remove file
#ifndef NDEBUG
    unlink(filename);
#endif

    // test H5::split_path separately
    std::vector<std::string> names(3);
    names[0] = "one";
    names[1] = "two";
    names[2] = "three";

    std::list<std::string> path = split_path("/one/two/three/");
    BOOST_CHECK(path.size() == 3);
    BOOST_CHECK(std::equal(path.begin(), path.end(), names.begin()));

    path = split_path("one/two/three");
    BOOST_CHECK(path.size() == 3);
    BOOST_CHECK(std::equal(path.begin(), path.end(), names.begin()));

    path = split_path("//one///two//three");
    BOOST_CHECK(path.size() == 3);
    BOOST_CHECK(std::equal(path.begin(), path.end(), names.begin()));
}
