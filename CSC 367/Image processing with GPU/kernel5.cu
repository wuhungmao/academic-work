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
#include <stdio.h>

void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time) {
  // This kernel inherit from kernel 4. Hence I will only put documentation at necessary places

  // There are two major optimizations
  // First optimization, instead of looping through output array to find global maximum
  // and minimum value, run_kernel5 invoke another kernel called global reduction
  // This greatly reduce amount of data required to be transfer. 

  // Second optimization, memory coalescing. Use the approach discussed in lecture to exploit 
  // memory coalescing. Applied to local reduction and global reduction.

  int num_pixels = width * height;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int num_of_thread_per_block = deviceProp.maxThreadsDim[0];
  int num_block = 32768;

  int chunk_per_thread = ((width*height / (num_of_thread_per_block*num_block)) == 0) ? 1: width*height / (num_of_thread_per_block*num_block);
  
  int *max_array_device;
  int *min_array_device;
  int8_t *filter_device;
  int32_t *input_device;
  int32_t *output_device;

  int *max_array_cpu;
  int *min_array_cpu;

  cudaError_t cudaStatus;
  size_t SIZE = num_pixels * sizeof(int32_t);
  size_t SIZE_filter = sizeof(int8_t) * dimension * dimension;
  size_t SIZE_max_min_array = sizeof(int)*num_block;

  cudaStatus = cudaMalloc((void**)&input_device, SIZE);
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaStatus = cudaMalloc((void**)&output_device, SIZE);
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
  }

  max_array_cpu = (int *) malloc(SIZE_max_min_array);
  min_array_cpu = (int *) malloc(SIZE_max_min_array);
  
  cudaStatus = cudaMalloc((void**)&max_array_device, SIZE_max_min_array); 
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
  }
  
  cudaStatus = cudaMalloc((void**)&min_array_device, SIZE_max_min_array); 
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaStatus = cudaMalloc((void**)&filter_device, SIZE_filter); 
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
  }

  //transfer in event
  cudaEvent_t transfer_in_start, transfer_in_stop;
  cudaEventCreate(&transfer_in_start);
  cudaEventCreate(&transfer_in_stop);
  cudaEventRecord(transfer_in_start);

  cudaStatus = cudaMemcpy(input_device, input, SIZE, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaStatus = cudaMemcpy(filter_device, filter, SIZE_filter, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) 
  {
      fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaEventRecord(transfer_in_stop);
  cudaEventSynchronize(transfer_in_stop);
  cudaEventElapsedTime(transfer_in_time, transfer_in_start, transfer_in_stop);

  //computation event merge to one
  cudaEvent_t computation_start_1, computation_stop_1;
  float computation_time_1;
  cudaEventCreate(&computation_start_1);
  cudaEventCreate(&computation_stop_1);
  cudaEventRecord(computation_start_1);

  kernel5<<<num_block, num_of_thread_per_block>>>(filter_device, dimension, input_device, output_device, width, height, max_array_device, min_array_device, chunk_per_thread);
  cudaDeviceSynchronize();

  //compute global max and global min using global_reduction kernel
  //If there were more than 1024 blocks created for computation, then number of block for this kernel 
  //is larger than 1. Otherwise create 1 block to prevent overhead
  int num_thread_per_block_global_reduction = num_block > num_of_thread_per_block ? num_of_thread_per_block : num_block;
  int num_block_global_reduction = num_block / num_of_thread_per_block == 0 ? 1 : (num_block + num_of_thread_per_block - 1)/ num_of_thread_per_block; 
  int size_of_max_array = num_block;

  global_reduction<<<num_block_global_reduction, num_thread_per_block_global_reduction>>>(max_array_device, min_array_device, size_of_max_array);
  cudaDeviceSynchronize();
  //Since we can now find global maximum and global minimum in gpu, we don't need to do it at cpu.
  //Hence, this way saves a transfer out and we don't need to divide computation and transfer out into two parts.
  normalize5<<<num_block, num_of_thread_per_block>>>(output_device, width, height, max_array_device, min_array_device, chunk_per_thread);
  cudaDeviceSynchronize();

  cudaEventRecord(computation_stop_1);
  cudaEventSynchronize(computation_stop_1);
  cudaEventElapsedTime(&computation_time_1, computation_start_1, computation_stop_1);

  cudaEvent_t transfer_out_start_1, transfer_out_stop_1;
  float transfer_out_time_1;
  cudaEventCreate(&transfer_out_start_1);
  cudaEventCreate(&transfer_out_stop_1);
  cudaEventRecord(transfer_out_start_1);

  cudaStatus = cudaMemcpy(output, output_device, SIZE, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaEventRecord(transfer_out_stop_1);
  cudaEventSynchronize(transfer_out_stop_1);
  cudaEventElapsedTime(&transfer_out_time_1, transfer_out_start_1, transfer_out_stop_1);

  *computation_time = computation_time_1;
  *transfer_out_time = transfer_out_time_1;

  //Don't forget cudafree
  cudaFree(input_device);
  cudaFree(output_device);
  cudaFree(max_array_device);
  cudaFree(min_array_device);
  cudaFree(filter_device);
  free(max_array_cpu);
  free(min_array_cpu);
}


/* Processes a single pixel and returns the value of processed pixel */
__device__ int32_t apply2d(const int8_t *f, int32_t dimension, const int32_t *input, int row, int column,
                int32_t width, int32_t height) 
{
  int val = 0;
  int filter_ind = 0;
  int offset = (dimension-1)/2;
  int start_row_ind = row-offset;
  int start_col_ind = column-offset;
  int end_row_ind = row-offset+dimension;
  int end_col_ind = column-offset+dimension;
  for (int i = start_row_ind; i < end_row_ind; i++) 
  {
    for (int j = start_col_ind; j < end_col_ind; j++) 
    {
      if(0 <= i && i < height && 0 <= j && j < width) {
        val += f[filter_ind] * input[i * width + j];
      }
      filter_ind++;
    }
  }
  return val;
}

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array, int chunk_per_thread) 
{
  int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  int local_tid = threadIdx.x;
  int start_pixel_ind = global_tid * chunk_per_thread;
  int end_pixel_ind = (global_tid+1) * chunk_per_thread;
  
  __shared__ int local_max_array[1024];
	__shared__ int local_min_array[1024];

  __shared__ int activeThreadCounter;

  if (threadIdx.x == 0) {
      activeThreadCounter = 0;
  }
  __syncthreads();

  //Find max and min value along with processing pixels
  int row_ind, col_ind, loc_max, loc_min;
  int local_array_ind = 0;
  loc_max = 0;
  loc_min = 0;

  //process through every pixel from start index to end index and store result in local_array
  for(int i = start_pixel_ind; i < end_pixel_ind && i < width * height; i++) 
  {
    col_ind = i%width;
    row_ind = i/width;
    output[i] = apply2d(filter, dimension, input, row_ind, col_ind, width, height);
    __syncthreads();
    if (output[i]> loc_max)
    {
      loc_max = output[i];
    }
    if (output[i] < loc_min)
    {
      loc_min = output[i];
    }
    local_array_ind++;
    __syncthreads();
  }


  if (start_pixel_ind < width * height) 
  {
    atomicAdd(&activeThreadCounter, 1);
    __syncthreads();  
    
    local_max_array[local_tid] = loc_max;
    local_min_array[local_tid] = loc_min;
    __syncthreads();
    //Exploit memory coalescing to reduce execution time
    for (unsigned int s = 1; s < activeThreadCounter; s *= 2) 
    {
      int idx = 2 * s * local_tid;
      if(idx < activeThreadCounter && local_max_array[idx] < local_max_array[idx + s] && idx + s < activeThreadCounter)  
      {
        local_max_array[idx] = local_max_array[idx + s];
      }
      else if(idx < activeThreadCounter && local_min_array[idx] > local_min_array[idx + s] && idx + s < activeThreadCounter)  
      {
        local_min_array[idx] = local_min_array[idx + s];
      }
      __syncthreads();
    }

    max_array[blockIdx.x] = local_max_array[0];
    min_array[blockIdx.x] = local_min_array[0];
    __syncthreads();
  }
}

/* This kernel is responsible for finding global maximum and global minimum */
__global__ void global_reduction(int *max_array, int *min_array, int size) {
   int tid = blockIdx.x*blockDim.x + threadIdx.x;
   for (int s = blockDim.x/2; s > 0; s >>= 1) 
   { 
      if (tid < s && max_array[tid] < max_array[tid + s] && tid + s < size)
      {    
         max_array[tid] = max_array[tid + s];
      }
      else if (tid < s && min_array[tid] > min_array[tid + s] && tid + s < size) 
      {
         min_array[tid] = min_array[tid + s];
      }
   __syncthreads();
   }
}

__device__ void normalize_pixel(int32_t *output, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) 
{
  if (smallest == largest) {
    return;
  }
  output[pixel_idx] =
      ((output[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/* Same access pattern as kernel 4 */
__global__ void normalize5(int32_t *output, int32_t width, int32_t height,
                           int32_t *max_array, int32_t *min_array, int chunk_per_thread) 
{
  int row_ind, col_ind, pixel_ind;
  int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  int start_pixel_ind = global_tid * chunk_per_thread;
  int end_pixel_ind = (global_tid+1) * chunk_per_thread;
  int global_min = min_array[0];
  int global_max = max_array[0];
  
  for(int i = start_pixel_ind; i < end_pixel_ind && i < width * height; i++) 
  {
    col_ind = i%width;
    row_ind = i/width;
    pixel_ind = row_ind * width + col_ind;
    normalize_pixel(output, pixel_ind, global_min, global_max);
  }
}
