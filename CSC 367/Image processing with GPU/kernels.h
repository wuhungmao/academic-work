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

#ifndef __KERNELS__H
#define __KERNELS__H

__device__ inline void find_pixel_pos(int arr[3*3][2], int dim, int col, int row, int width, int height);
__device__ inline int32_t apply2d(const int8_t *f, int32_t dimension, const int32_t *input, int row, int column,
                int32_t width, int32_t height);
__device__ inline void normalize_pixel(int32_t *output, int32_t pixel_idx, int32_t smallest,
                     int32_t largest);

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height);

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time);

__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array); 

__global__ void normalize1(int32_t *output, int32_t width, int32_t height,
                           int32_t global_max, int32_t global_min);

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time);
__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array);

__global__ void normalize2(int32_t *output, int32_t width, int32_t height,
                           int32_t global_max, int32_t global_min);

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time);
                 
__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array, int num_chunk_per_thread);

__global__ void normalize3(int32_t *output, int32_t width, int32_t height,
                           int32_t global_max, int32_t global_min, int chunk_per_thread);

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time);
__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array, int stride_size);
__global__ void normalize4(int32_t *output, int32_t width, int32_t height,
                           int32_t global_max, int32_t global_min, int stride_size);

void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float * transfer_in_time, float *computation_time ,float *transfer_out_time);

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int *max_array, int *min_array, int stride_size);

__global__ void normalize5(int32_t *output, int32_t width, int32_t height,
                           int32_t *max_array, int32_t *min_array, int stride_size);

__global__ void global_reduction(int *max_array, int *min_array, int size);


#endif
