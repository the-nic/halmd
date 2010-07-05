/*
 * Copyright © 2010  Peter Colberg and Felix Höfling
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

#include <halmd/io/logger.hpp>
#include <halmd/random/gpu/rand48.hpp>
#include <halmd/random/gpu/random.hpp>
#include <halmd/random/gpu/random_kernel.hpp>

namespace halmd
{
namespace random { namespace gpu
{

/**
 * Assemble module options
 */
template <typename RandomNumberGenerator>
void random<RandomNumberGenerator>::options(po::options_description& desc)
{
    po::options_description group("Random number generator (GPU)");
    group.add_options()
        ("random-threads", po::value<unsigned int>()->default_value(32 << DEVICE_SCALE),
         "number of CUDA threads per block")
        ("random-blocks", po::value<unsigned int>()->default_value(32),
         "number of CUDA blocks")
        ;
    desc.add(group);
}

/**
 * Resolve module dependencies
 */
template <typename RandomNumberGenerator>
void random<RandomNumberGenerator>::depends()
{
    modules::depends<_Self, device_type>::required();
}

template <typename RandomNumberGenerator>
random<RandomNumberGenerator>::random(modules::factory& factory, po::options const& vm)
  : _Base(factory, vm)
  // dependency injection
  , device(modules::fetch<device_type>(factory, vm))
  // allocate random number generator state
  , rng_(vm["random-blocks"].as<unsigned int>(), vm["random-threads"].as<unsigned int>())
{
    _Base::seed(vm);
}

template <typename RandomNumberGenerator>
void random<RandomNumberGenerator>::seed(unsigned int value)
{
    LOG("random number generator seed: " << value);

    try {
        rng_.seed(value);
        cuda::copy(rng_.rng(), random_wrapper<rng_type>::rng);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to seed random number generator");
    }
}

/**
 * fill array with uniform random numbers in [0.0, 1.0)
 */
template <typename RandomNumberGenerator>
void random<RandomNumberGenerator>::uniform(cuda::vector<float>& g_v)
{
    try {
        cuda::configure(rng_.dim.grid, rng_.dim.block);
        random_wrapper<rng_type>::uniform(g_v, g_v.size());
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to fill vector with uniform random numbers");
    }
}

/**
 * fill array with random integers in [0, 2^32-1]
 */
template <typename RandomNumberGenerator>
void random<RandomNumberGenerator>::get(cuda::vector<unsigned int>& g_v)
{
    try {
        cuda::configure(rng_.dim.grid, rng_.dim.block);
        random_wrapper<rng_type>::get(g_v, g_v.size());
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to fill vector with uniform integer random numbers");
    }
}

/**
 * fill array with normal distributed random numbers in [0.0, 1.0)
 */
template <typename RandomNumberGenerator>
void random<RandomNumberGenerator>::normal(cuda::vector<float>& g_v, float mean, float sigma)
{
    try {
        cuda::configure(rng_.dim.grid, rng_.dim.block);
        random_wrapper<rng_type>::normal(g_v, g_v.size(), mean, sigma);
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to fill vector with normal random numbers");
    }
}

}} // namespace random::gpu

template class random::gpu::random<random::gpu::rand48>;

template class module<random::gpu::random<random::gpu::rand48> >;

} // namespace halmd
