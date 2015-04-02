/*
 * Copyright © 2014 Nicolas Höft
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

#include <halmd/config.hpp>

#define BOOST_TEST_MODULE binning
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <cmath>

#include <halmd/mdsim/host/region.cpp> // include definition file in order to generate template instatiations of region with 'simple' geometry
#include <halmd/mdsim/positions/lattice_primitive.hpp>
#include <test/tools/ctest.hpp>
#include <test/tools/init.hpp>
#ifdef HALMD_WITH_GPU
# include <halmd/mdsim/gpu/region.cpp>
# include <test/tools/cuda.hpp>
#endif

#include <test/unit/mdsim/geometries/simple.hpp>

/**
 * Test particle regions.
 */
template <typename region_type, typename geometry_type>
static void
test_region(region_type& region, geometry_type& geometry, typename region_type::particle_type const& particle, typename region_type::box_type const& box)
{
    typedef typename region_type::particle_type particle_type;
    typedef typename region_type::box_type box_type;
    typedef typename region_type::vector_type vector_type;
    typedef typename region_type::array_type array_type;
    typedef typename region_type::size_type size_type;

    // get particle positions
    std::vector<vector_type> position;
    position.reserve(particle.nparticle());
    get_position(particle, back_inserter(position));

    // allocate output array for indices of particles
    std::vector<unsigned int> particle_index;
    particle_index.reserve(particle.nparticle());

    // setup profiling timer and connect to profiler
    // (we have no access to the private struct region.runtime)
    halmd::utility::profiler profiler;
    std::shared_ptr<halmd::accumulator<double>> runtime = std::make_shared<halmd::accumulator<double>>();
    profiler.on_profile(runtime, "update particle region");

    typedef typename region_type::iterator_range_type iterator_range_type;
    iterator_range_type included_range;
    iterator_range_type excluded_range;
    {
        halmd::utility::profiler::scoped_timer_type timer(*runtime);
        excluded_range = region.excluded();
        included_range = region.included();
    }
    // output profiling timers
    profiler.profile();

    // test if the mask is correct
    std::vector<size_type> mask;
    mask.reserve(particle.nparticle());
    get_mask(region, back_inserter(mask));
    for(int i = 0; i < particle.nparticle(); ++i) {
        vector_type r = position[i];
        box.reduce_periodic(r);
        size_type mask_expected = geometry(r) ? 1 : 0;
        BOOST_CHECK_EQUAL(mask_expected, mask[i]);
    }

    get_excluded(region, back_inserter(particle_index));
    get_included(region, back_inserter(particle_index));

    // check that each particle is sorted into the correct bin
    for (int i = 0; i < region.nexcluded(); ++i) {
        unsigned int idx = particle_index[i];
        vector_type r = position[idx];
        box.reduce_periodic(r);
        BOOST_CHECK_EQUAL(geometry(r), false);
    }
    for (int i = region.nexcluded(); i < particle.nparticle(); ++i) {
        unsigned int idx = particle_index[i];
        vector_type r = position[idx];
        box.reduce_periodic(r);
        BOOST_CHECK_EQUAL(geometry(r), true);
    }

    // check that all particles were binned (inside/outside region)
    sort(particle_index.begin(), particle_index.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        particle_index.begin()
      , particle_index.end()
      , boost::counting_iterator<unsigned int>(0)
      , boost::counting_iterator<unsigned int>(particle.nparticle())
    );
}

template<typename region_type, typename shape_type, typename geometry_type>
static void
test_uniform_density(shape_type const& shape, std::shared_ptr<geometry_type>& geometry)
{
    enum { dimension = geometry_type::vector_type::static_size };
    typedef typename region_type::vector_type vector_type;
    typedef typename vector_type::value_type float_type;
    typedef typename region_type::particle_type particle_type;
    typedef typename region_type::box_type box_type;
    typedef typename shape_type::value_type size_type;

    // convert box edge lengths to edge vectors
    boost::numeric::ublas::diagonal_matrix<typename box_type::matrix_type::value_type> edges(shape.size());
    for (unsigned int i = 0; i < shape.size(); ++i) {
        edges(i, i) = shape[i];
    }

    // create close-packed lattice of given shape
    halmd::close_packed_lattice<vector_type, shape_type> lattice(shape);
    // create simulation domain
    std::shared_ptr<box_type> box(new box_type(edges));
    // create system of particles of number of lattice points
    std::shared_ptr<particle_type> particle(new particle_type(lattice.size(), 1));
    // create particle region
    region_type region(particle, box, geometry);

    BOOST_TEST_MESSAGE( "number density " << particle->nparticle() / box->volume() );

    // place particles on lattice
    set_position(
        *particle
      , boost::make_transform_iterator(boost::make_counting_iterator(size_type(0)), lattice)
    );

    // bin particles and test output
    test_region(region, *geometry, *particle, *box);
}

/**
 * Manual test case registration.
 */
HALMD_TEST_INIT( region )
{
    using namespace boost::unit_test;

    test_suite* ts_host = BOOST_TEST_SUITE( "host" );
    framework::master_test_suite().add(ts_host);

    test_suite* ts_host_two = BOOST_TEST_SUITE( "two" );
    ts_host->add(ts_host_two);

    test_suite* ts_host_three = BOOST_TEST_SUITE( "three" );
    ts_host->add(ts_host_three);

#ifdef HALMD_WITH_GPU
    test_suite* ts_gpu = BOOST_TEST_SUITE( "gpu" );
    framework::master_test_suite().add(ts_gpu);

    test_suite* ts_gpu_two = BOOST_TEST_SUITE( "two" );
    ts_gpu->add(ts_gpu_two);

    test_suite* ts_gpu_three = BOOST_TEST_SUITE( "three" );
    ts_gpu->add(ts_gpu_three);
#endif

    {
        int constexpr dimension = 2;
        typedef double float_type;
        typedef simple_geometry<dimension, float_type> geometry_type;
        typedef halmd::mdsim::host::region<dimension, float_type, geometry_type> region_type;
        typedef halmd::fixed_vector<size_t, dimension> shape_type;

        auto uniform_density = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({0,0}));
            test_uniform_density<region_type, shape_type>(
                {4, 7} // non-square box
              , geometry
            );
        };
        ts_host_two->add(BOOST_TEST_CASE( uniform_density ));
    }
    {
        int constexpr dimension = 3;
        typedef double float_type;
        typedef simple_geometry<dimension, float_type> geometry_type;
        typedef halmd::mdsim::host::region<dimension, float_type, geometry_type> region_type;
        typedef halmd::fixed_vector<size_t, dimension> shape_type;

        auto uniform_density = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({0,0,0}));
            test_uniform_density<region_type, shape_type>(
                {40, 70, 90} // non-square box
              , geometry
            );
        };
        ts_host_three->add(BOOST_TEST_CASE( uniform_density ));
    }
#ifdef HALMD_WITH_GPU
    {
        int constexpr dimension = 2;
        typedef float float_type;
        typedef simple_geometry<dimension, float_type> geometry_type;
        typedef halmd::mdsim::gpu::region<dimension, float_type, geometry_type> region_type;
        typedef halmd::fixed_vector<size_t, dimension> shape_type;

        auto uniform_density = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({0,0}));
            test_uniform_density<region_type, shape_type>(
                {4, 7} // non-square box
              , geometry
            );
        };
        ts_gpu_two->add(BOOST_TEST_CASE( uniform_density ));

        auto uniform_density_all_excluded = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({-4,-7}));
            test_uniform_density<region_type, shape_type>(
                {4, 7} // non-square box with coprime edge lengths
              , geometry
            );
        };
        ts_gpu_two->add(BOOST_TEST_CASE( uniform_density_all_excluded ));

        auto uniform_density_all_included = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({4,7}));
            test_uniform_density<region_type, shape_type>(
                {4, 7} // non-square box
              , geometry
            );
        };
        ts_gpu_two->add(BOOST_TEST_CASE( uniform_density_all_included ));
    }
    {
        int constexpr dimension = 3;
        typedef float float_type;
        typedef simple_geometry<dimension, float_type> geometry_type;
        typedef halmd::mdsim::gpu::region<dimension, float_type, geometry_type> region_type;
        typedef halmd::fixed_vector<size_t, dimension> shape_type;

        auto uniform_density = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({0,0,0}));
            test_uniform_density<region_type, shape_type>(
                {40, 70, 90}
              , geometry
            );
        };
        ts_gpu_three->add(BOOST_TEST_CASE( uniform_density ));

        auto uniform_density_all_excluded = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({-4,-7,-9}));
            test_uniform_density<region_type, shape_type>(
                {4, 7, 9}
              , geometry
            );
        };
        ts_gpu_three->add(BOOST_TEST_CASE( uniform_density_all_excluded ));

        auto uniform_density_all_included = [=]() {
            std::shared_ptr<geometry_type> geometry(new geometry_type({4,7,9}));
            test_uniform_density<region_type, shape_type>(
                {4, 7, 9}
              , geometry
            );
        };
        ts_gpu_three->add(BOOST_TEST_CASE( uniform_density_all_included ));
    }
#endif
}
