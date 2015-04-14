/*
 * Copyright © 2015 Nicolas Höft
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

#define BOOST_TEST_MODULE copy_if

#include <algorithm>

#include <boost/test/unit_test.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <halmd/algorithm/gpu/copy_if.hpp>
#include <halmd/utility/timer.hpp>
#include <test/tools/cuda.hpp>
#include <test/tools/ctest.hpp>
#include <test/tools/init.hpp>
#include <test/unit/algorithm/gpu/copy_if_kernel.hpp>

BOOST_GLOBAL_FIXTURE( set_cuda_device )

template <typename T, typename Predicate>
static void test_copy_if(cuda::host::vector<T> const& h_v, Predicate pred)
{
    BOOST_TEST_MESSAGE("Processing array with " << h_v.size() << " elements");
    cuda::vector<T> g_v_in(h_v.size());
    cuda::vector<T> g_v_out(h_v.size());
    cuda::copy(h_v.begin(), h_v.end(), g_v_in.begin());

    auto last_output = halmd::copy_if(g_v_in.begin(), g_v_in.end(), g_v_out.begin(), pred);
    std::size_t out_len = last_output - g_v_out.begin();

    BOOST_TEST_MESSAGE("CUDA copied " << out_len << " elements");
    cuda::host::vector<T> h_out(out_len);
    cuda::copy(g_v_out.begin(), g_v_out.begin() + out_len, h_out.begin());

    std::vector<T> h_reference(h_v.size());
    auto last_ref = std::copy_if(h_v.begin(), h_v.end(), h_reference.begin(), pred);
    h_reference.resize(last_ref - h_reference.begin());

    BOOST_TEST_MESSAGE( "CPU copied " << h_reference.size() << " elements");

    // compare results
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_out.begin()
      , h_out.end()
      , h_reference.begin()
      , h_reference.end()
    );
}

BOOST_AUTO_TEST_CASE( copy_integer )
{
    cuda::host::vector<int> h_v(
        boost::make_counting_iterator(0)
      , boost::make_counting_iterator(12345679)
    );
    BOOST_TEST_MESSAGE("select_all");
    test_copy_if(h_v, select_all<int>());
    BOOST_TEST_MESSAGE("select_none");
    test_copy_if(h_v, select_none<int>());
    BOOST_TEST_MESSAGE("select_even");
    test_copy_if(h_v, select_even<int>());
    BOOST_TEST_MESSAGE("select_odd");
    test_copy_if(h_v, select_odd<int>());
}
