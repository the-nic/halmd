/* Parallel radix sort
 *
 * Copyright (C) 2008  Peter Colberg
 *
 * This program is free software: you can redistribute it and/or modify
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

#ifndef MDSIM_RADIX_HPP
#define MDSIM_RADIX_HPP

#include <algorithm>
#include <boost/ref.hpp>
#include "gpu/radix_glue.hpp"
#include "scan.hpp"

namespace mdsim
{

/*
 * Parallel radix sort
 */
template <typename T>
class radix_sort
{
public:
    radix_sort() : blocks(0), threads(0) {}

    /**
     * allocate parallel radix sort for given element count
     */
    radix_sort(uint const& count, uint const& blocks, uint const& threads) : blocks(blocks), threads(threads), g_bucket(bucket_size()), g_key(count), g_val(count), g_scan(scan_size(), scan_threads()) {}

    /**
     * reallocate parallel radix sort for given element count
     */
    void resize(uint const& count, uint const& blocks, uint const& threads)
    {
	radix_sort temp(count, blocks, threads);
	swap(temp);
    }

    /**
     * swap dimensions and device memory with another parallel radix sort
     */
    void swap(radix_sort& r)
    {
	std::swap(blocks, r.blocks);
	std::swap(threads, r.threads);
	g_bucket.swap(r.g_bucket);
	g_key.swap(r.g_key);
	g_val.swap(r.g_val);
	g_scan.swap(r.g_scan);
    }

    /**
     * radix sort given keys and values in-place
     */
    void operator()(cuda::vector<uint>& g_key_, cuda::vector<T>& g_val_, cuda::stream& stream)
    {
	assert(g_key_.size() == g_key.size());
	assert(g_val_.size() == g_val.size());

	// assign GPU dual buffers, as in the CUDA SDK radix sort example
	boost::reference_wrapper<cuda::vector<uint> > key[2] = { boost::ref(g_key_), boost::ref(g_key) };
	boost::reference_wrapper<cuda::vector<T> > val[2] = { boost::ref(g_val_), boost::ref(g_val) };

	for (uint r = 0; r < 32; r += gpu::radix::RADIX) {
	    // compute partial radix counts
	    cuda::configure(blocks, threads, shared_mem(), stream);
	    gpu::radix::histogram_keys(key[0].get(), g_bucket, g_key.size(), r);

	    // parallel prefix sum over radix counts
	    g_scan(g_bucket, stream);

	    // permute array
	    cuda::configure(blocks, threads, shared_mem(), stream);
	    gpu::radix::permute(key[0].get(), key[1].get(), val[0].get(), val[1].get(), g_bucket, g_key.size(), r);

	    // swap GPU dual buffers
	    std::swap(key[0], key[1]);
	    std::swap(val[0], val[1]);
	}
    }

private:
    /**
     * returns shared memory size in bytes for GPU radix kernels
     */
    uint shared_mem()
    {
	return threads * gpu::radix::BUCKETS_PER_THREAD * sizeof(uint);
    }

    /**
     * returns element count for GPU prefix sum
     */
    uint scan_size()
    {
	return blocks * threads * gpu::radix::BUCKETS_PER_THREAD;
    }

    /**
     * returns number of CUDA threads per block for GPU prefix sum
     */
    uint scan_threads()
    {
	return gpu::radix::BUCKET_SIZE;
    }

    /**
     * returns radix count bucket element count
     */
    uint bucket_size()
    {
	return blocks * threads * gpu::radix::BUCKETS_PER_THREAD;
    }

private:
    uint blocks, threads;
    cuda::vector<uint> g_bucket, g_key;
    cuda::vector<T> g_val;
    prefix_sum<uint> g_scan;
};

} // namespace mdsim

#endif /* ! MDSIM_RADIX_HPP */
