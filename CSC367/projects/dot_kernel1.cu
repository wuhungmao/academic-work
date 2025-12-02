/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2022 Bogdan Simion
 * -------------
 */

#include "kernels.h"

__global__ void dot_kernel1(float *g_idata1, float *g_idata2, float *g_odata) {
    
	extern __shared__ float sdata[];
	
	/* This is not a global tid, it's the tid within the block.
	Since the compiler creates a copy of sdata per block, then 
	to index into the sdata array we only need a thread's id within 
	its block. */
	unsigned int tid = threadIdx.x;
	
	// Global thread id
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = g_idata1[i] * g_idata2[i];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0)  { // In a warp, only thread ids divisible by the step participate
			sdata[tid] += sdata[tid + s];
		} 
		__syncthreads();
	}

	// write result for this block back to global memory
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
