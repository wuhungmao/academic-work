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

__global__ void dot_kernel5(float *g_idata1, float *g_idata2, float *g_odata) {
    
	extern __shared__ float sdata[];
    unsigned int blockSize = blockDim.x;
	unsigned int tid = threadIdx.x;
	
	// Global thread id
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

	sdata[tid] = g_idata1[i] * g_idata2[i] + g_idata1[i + blockDim.x] * g_idata2[i + blockDim.x];
	__syncthreads();

	// do reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) { 
        if (tid < s) {  
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

	if (tid < 32) {
		volatile float* smem = sdata;
		if (blockSize >= 64) smem[tid] += smem[tid + 32];
		if (blockSize >= 32) smem[tid] += smem[tid + 16];
		if (blockSize >= 16) smem[tid] += smem[tid + 8];
		if (blockSize >=  8) smem[tid] += smem[tid + 4];
		if (blockSize >=  4) smem[tid] += smem[tid + 2];
		if (blockSize >=  2) smem[tid] += smem[tid + 1];
	} 

	// write result for this block back to global memory
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}