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

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time) {
  int num_pixels = width * height;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int num_of_thread_per_block = deviceProp.maxThreadsDim[0];
  int num_block = 32768;

  //stride over total number of threads
  int stride_size = num_block * num_of_thread_per_block;

  //each block stores 2 values local max and local min on max_array and min_array
  int *max_array_device;
  int *min_array_device;
  int8_t *filter_device;
  int32_t *input_device;
  int32_t *output_device;

  int *max_array_cpu;
  int *min_array_cpu;

  //cudamalloc to malloc memory on device memory. 
  cudaError_t cudaStatus;
  size_t SIZE = num_pixels * sizeof(int32_t);
  size_t SIZE_filter = sizeof(int8_t) * dimension * dimension;
  size_t SIZE_max_min_array = sizeof(int)*num_block;

  int global_max_cpu;
  int global_min_cpu;

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

  //computation event 
  cudaEvent_t computation_start_1, computation_stop_1;
  float computation_time_1;
  cudaEventCreate(&computation_start_1);
  cudaEventCreate(&computation_stop_1);
  cudaEventRecord(computation_start_1);

  kernel4<<<num_block, num_of_thread_per_block>>>(filter_device, dimension, input_device, output_device, width, height, max_array_device, min_array_device, stride_size);
  cudaDeviceSynchronize();

  cudaEventRecord(computation_stop_1);
  cudaEventSynchronize(computation_stop_1);
  cudaEventElapsedTime(&computation_time_1, computation_start_1, computation_stop_1);

  //transfer out event
  //Copied computed result stored in output array to output on cpu
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

  cudaEvent_t computation_start_2, computation_stop_2;
  float computation_time_2;
  cudaEventCreate(&computation_start_2);
  cudaEventCreate(&computation_stop_2);
  cudaEventRecord(computation_start_2);
  
  //compute global max and global min
  global_max_cpu = max_array_cpu[0];
  global_min_cpu = min_array_cpu[0];
  for (int i = 0; i < num_block; i++) {
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
  normalize4<<<num_block, num_of_thread_per_block>>>(output_device, width, height, global_max_cpu, global_min_cpu, stride_size);
  cudaEventRecord(computation_stop_2);
  cudaEventSynchronize(computation_stop_2);
  cudaEventElapsedTime(&computation_time_2, computation_start_2, computation_stop_2);

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

  *computation_time = computation_time_1 + computation_time_2;
  *transfer_out_time = transfer_out_time_1 + transfer_out_time_2;

  //Free allocated space
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

/* Process pixels with stride */
__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array, int stride_size) 
{
  int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  int local_tid = threadIdx.x;
  int start_pixel_ind = global_tid;

  //Similar to kernel 3
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
  //process through every pixel from start index and store result in local_array
  for(int i = start_pixel_ind; i < width * height; i+=stride_size) 
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

    //store loc_max and loc_min into local_max_array and local_min_array
    local_max_array[local_tid] = loc_max;
    local_min_array[local_tid] = loc_min;
    __syncthreads();

    //Each thread inside a block participate in this reduction
    //process to find out maximum and minimum value of all pixels
    //computed by every thread of the block 
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
    
    max_array[blockIdx.x] = local_max_array[0];
    min_array[blockIdx.x] = local_min_array[0];
    __syncthreads();

  }

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

/* Every thread normalize the chunk of pixel assigned to it. */
__global__ void normalize4(int32_t *output, int32_t width, int32_t height,
                           int32_t global_max, int32_t global_min, int stride_size) 
{
  int row_ind, col_ind, pixel_ind;
  int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  int start_pixel_ind = global_tid;

  for(int i = start_pixel_ind; i < width * height; i+=stride_size) 
  {
    col_ind = i%width;
    row_ind = i/width;
    pixel_ind = row_ind * width + col_ind;
    normalize_pixel(output, pixel_ind, global_min, global_max);
  }
}
