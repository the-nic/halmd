/*
 * Copyright © 2007-2010  Peter Colberg
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

#ifndef HALMD_RANDOM_GPU_RAND48_KERNEL_HPP
#define HALMD_RANDOM_GPU_RAND48_KERNEL_HPP

#include <cuda_wrapper.hpp>
#include <halmd/random/gpu/rand48_kernel.cuh>

namespace halmd
{
namespace random { namespace gpu
{

struct rand48_wrapper
{
    static cuda::function<void (uint48*)> leapfrog;
    static cuda::function<void (uint48 const*, uint48 const*, uint48*, uint48*, ushort3* g_state, uint)> seed;
};

}} // namespace random::gpu

} // namespace halmd

#endif /* ! HALMD_RANDOM_GPU_RAND48_KERNEL_HPP */
