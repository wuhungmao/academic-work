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

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time) {  
  int total_num_threads, num_pixels;
  num_pixels = total_num_threads = width * height;

  //Every block created will have 1024 threads. Minimum number of block is 1.
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int num_of_thread_per_block = deviceProp.maxThreadsDim[0];
  int num_block = (total_num_threads/num_of_thread_per_block<=0) ? 1 : (total_num_threads+num_of_thread_per_block-1)/num_of_thread_per_block;
  //These are all variables and array passed to a kernel, they will be allocated on gpu memory 
  int *max_array_device;
  int *min_array_device;
  int8_t *filter_device;
  int32_t *input_device;
  int32_t *output_device;

  //each block stores 2 values local max and local min on max_array and min_array.
  int *max_array_cpu;
  int *min_array_cpu;

  //cudamalloc to malloc memory on device memory. 
  cudaError_t cudaStatus;
  size_t SIZE = num_pixels * sizeof(int32_t);
  size_t SIZE_filter = sizeof(int8_t) * dimension * dimension;
  size_t SIZE_max_min_array = sizeof(int)*num_block;

  int global_max_cpu;
  int global_min_cpu;
  
  //Allocate space on gpu memory for device variable and array
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
  //input_device and filter_device array will to contain value from input and filter.
  //So kernel can use these value stored on gpu

  //calculate transfer in time
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

  //computation event 
  //Process is very similar to kernel 1
  cudaEvent_t computation_start_1, computation_stop_1;
  float computation_time_1;
  cudaEventCreate(&computation_start_1);
  cudaEventCreate(&computation_stop_1);
  cudaEventRecord(computation_start_1);

  kernel2<<<num_block, num_of_thread_per_block>>>(filter_device, dimension, input_device, output_device, width, height, max_array_device, min_array_device);
  cudaDeviceSynchronize();

  cudaEventRecord(computation_stop_1);
  cudaEventSynchronize(computation_stop_1);
  cudaEventElapsedTime(&computation_time_1, computation_start_1, computation_stop_1);

  //transfer out event
  //Copied computed result stored in output array to output on cpu
  //Similar to computation event, this event is divided into two parts as well
  cudaEvent_t transfer_out_start_1, transfer_out_stop_1;
  float transfer_out_time_1;
  cudaEventCreate(&transfer_out_start_1);
  cudaEventCreate(&transfer_out_stop_1);
  cudaEventRecord(transfer_out_start_1);

  cudaStatus = cudaMemcpy(max_array_cpu, max_array_device, sizeof(int)*num_block, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
  }
  cudaStatus = cudaMemcpy(min_array_cpu, min_array_device, sizeof(int)*num_block, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaEventRecord(transfer_out_stop_1);
  cudaEventSynchronize(transfer_out_stop_1);
  cudaEventElapsedTime(&transfer_out_time_1, transfer_out_start_1, transfer_out_stop_1);

  //Second part of computation. Finding global max and min and normalization.
  cudaEvent_t computation_start_2, computation_stop_2;
  float computation_time_2;
  cudaEventCreate(&computation_start_2);
  cudaEventCreate(&computation_stop_2);
  cudaEventRecord(computation_start_2);

  //compute global max and global min
  global_max_cpu = max_array_cpu[0];
  global_min_cpu = min_array_cpu[0];
  for (int i = 0; i < num_block; i++) 
  {
    if (max_array_cpu[i]>global_max_cpu) 
    {
      global_max_cpu = max_array_cpu[i];
    }
    if (min_array_cpu[i]<global_min_cpu) 
    {
      global_min_cpu = min_array_cpu[i];
    }
  }

  //normalize using global max cpu and global min cpu
  normalize2<<<num_block, num_of_thread_per_block>>>(output_device, width, height, global_max_cpu, global_min_cpu);
  cudaDeviceSynchronize();
  cudaEventRecord(computation_stop_2);
  cudaEventSynchronize(computation_stop_2);
  cudaEventElapsedTime(&computation_time_2, computation_start_2, computation_stop_2);

  //Transfer out final result to output
  cudaEvent_t transfer_out_start_2, transfer_out_stop_2;
  float transfer_out_time_2;
  cudaEventCreate(&transfer_out_start_2);
  cudaEventCreate(&transfer_out_stop_2);
  cudaEventRecord(transfer_out_start_2);

  cudaStatus = cudaMemcpy(output, output_device, SIZE, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy fail: %s\n", cudaGetErrorString(cudaStatus));
  }

  cudaEventRecord(transfer_out_stop_2);
  cudaEventSynchronize(transfer_out_stop_2);
  cudaEventElapsedTime(&transfer_out_time_2, transfer_out_start_2, transfer_out_stop_2);

  //Add up computation time and transfer out time
  *computation_time = computation_time_1 + computation_time_2;
  *transfer_out_time = transfer_out_time_1 + transfer_out_time_2;

  //Free gpu and cpu memory
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

/* Utilize multiple threads to do computation */
__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array) {
  
  int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  int local_tid = threadIdx.x;
	__shared__ int local_max_array[1024];
	__shared__ int local_min_array[1024];

  //need to know number of active thread
  __shared__ int activeThreadCounter;

  if (threadIdx.x == 0) {
      activeThreadCounter = 0;
  }
  __syncthreads();

  //Calculate total number of threads which should actually do work 
  if(global_tid < width*height) {
      atomicAdd(&activeThreadCounter, 1);
  }
  __syncthreads();  
  
  //Notice this kernel is row major 
  int row_ind, col_ind;
  col_ind = global_tid%width;
  row_ind = global_tid/width;

  int32_t processed_pixel = apply2d(filter, dimension, input, row_ind, col_ind, width, height);
  __syncthreads();

  if (global_tid < width * height) 
  {
    output[row_ind * width + col_ind] = processed_pixel;
  }
  local_max_array[local_tid] = processed_pixel;
  local_min_array[local_tid] = processed_pixel;
  __syncthreads();

  //Start local reduction on 1024 values or less in a block
  //and store max and min value within a block in max_min_array 
  //based on block id
	for (unsigned int s = 1; s < activeThreadCounter; s *= 2) 
  {
		if(local_tid % (2*s) == 0 && local_max_array[local_tid] < local_max_array[local_tid + s] && local_tid + s < activeThreadCounter)  
    { 
      local_max_array[local_tid] = local_max_array[local_tid + s];
    }
    else if(local_tid % (2*s) == 0 && local_min_array[local_tid] > local_min_array[local_tid + s] && local_tid + s < activeThreadCounter)  
    {
      local_min_array[local_tid] = local_min_array[local_tid + s];
		}
		__syncthreads();
	}

  //local_max_array and local_min_array have local max 
  //and local min at index 0 respectively
  max_array[blockIdx.x] = local_max_array[0];
  min_array[blockIdx.x] = local_min_array[0];
  __syncthreads();
}

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
__device__ void normalize_pixel(int32_t *output, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) 
{
  if (smallest == largest) {
    return;
  }
  output[pixel_idx] =
      ((output[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/* Similar to kernel1, but access pattern is row major */
__global__ void normalize2(int32_t *output, int32_t width, int32_t height,
                           int32_t global_max, int32_t global_min) 
{
  int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  int row_ind, col_ind;
  col_ind = global_tid%width;
  row_ind = global_tid/width;
  int pixel_ind = row_ind * width + col_ind;
  normalize_pixel(output, pixel_ind, global_min, global_max);
}
