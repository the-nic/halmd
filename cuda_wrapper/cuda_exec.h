/* cuda_exec.h
 *
 * Copyright (C) 2007  Peter Colberg
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

#ifndef __CUDA_EXEC_H__
#define __CUDA_EXEC_H__

#include "cuda_base.h"
#include "cuda_error.h"

#ifndef __CUDACC__

/*
 * CUDA execution configuration
 */
class cuda_dim
{
public:
  /* grid dimensions */
  const dim3 grid;
  /* block dimensions */
  const dim3 block;
  /* FIXME store useful numbers (no. of threads per grid/block) */

  cuda_dim(dim3 grid, dim3 block) : grid(grid), block(block)
  {
    /* FIXME store useful numbers (no. of threads per grid/block) */
  }

  int threads() const
  {
    return grid.y * grid.x * block.z * block.y * block.x;
  }

  int blocks_per_grid() const
  {
    return grid.y * grid.x;
  }

  int threads_per_block() const
  {
    return block.z * block.y * block.x;
  }
};


/*
 * CUDA execution control
 */

struct cuda_exec
{
    /*
     * configure execution parameters
     */
    static void configure(const cuda_dim& dim, size_t shared_mem = 0)
    {
	CUDA_CALL(cudaConfigureCall(dim.grid, dim.block, shared_mem, 0));
    }

    /*
     * push arbitrary argument into argument passing area
     */
    template <typename T>
    static void setup_argument(const T& arg, size_t *offset)
    {
	/* respect alignment requirements of passed argument */
	if (0 != *offset % __alignof(T)) {
	    *offset += __alignof(T) - *offset % __alignof(T);
	}

	CUDA_CALL(cudaSetupArgument(&arg, sizeof(T), *offset));

	/* advance argument offset for next call */
	*offset += sizeof(T);
    }

    /*
     * launch kernel
     */
    template <typename T>
    static void launch(T *entry)
    {
	CUDA_CALL(cudaLaunch(reinterpret_cast<const char *>(entry)));
    }
};

#endif /* ! __CUDACC__ */

#endif /* ! __CUDA_EXEC_H__ */
