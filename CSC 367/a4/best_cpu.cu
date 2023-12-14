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
#include <pthread.h>
#include <iostream>
#include <thread>

typedef struct work_t work;

struct common_work_t; 

typedef struct filter_t {
  int32_t dimension;
  const int8_t *matrix;
} filter;

typedef struct work_t
{
  common_work_t *common;
  int32_t id;
  int32_t assigned;
  int32_t num_pixels;
  int32_t start_pos[2];
  int32_t end_pos[2];
  int32_t method;
} work;

typedef struct common_work_t
{
  filter *f;
  const int32_t *original_image;
  int32_t *output_image;
  int32_t width;
  int32_t height;
  int32_t max_threads;
  int32_t total_work;
  int32_t* local_max_min_arr;
  pthread_barrier_t barrier;
} common_work;

pthread_mutex_t normalizable_mutex;
pthread_mutex_t target_mutex;
pthread_mutex_t work_pool_mutex;
pthread_mutex_t debug_mutex;

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel_cpu(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) 
{
  if (smallest == largest) {
    return;
  }
  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/* find out all locals for every pixel assigned to a block */
void find_pixel_pos_cpu(int (&arr)[][2], int dim, int col, int row, int width, int height) 
{
  int offset = (dim-1)/2;
  int arr_ind = 0;
  for (int row_ind = -offset; row_ind < offset+1; row_ind++) {
    for (int col_ind = -offset; col_ind < (offset+1); col_ind++){
      //find all positions of pixels to which a filter is applied
      if (col+col_ind >= 0 && row+row_ind >=0 && col + col_ind < width && row + row_ind < height){
        arr[arr_ind][0] = row+row_ind;
        arr[arr_ind][1] = col+col_ind;
      } else {
        arr[arr_ind][0] = -1;
        arr[arr_ind][1] = -1;
      }
      arr_ind++;
    }
  }
}

/* calculate maximum and minimum value in a array */
void find_max_and_min(int target[], int size, int& max, int& min) 
{
  max = target[0];
  min = target[0];
  for (int i = 0; i < size; i++){
    if (max < target[i]) {
      max = target[i];
    } 
    if (min > target[i]) {
      min = target[i];
    }
  }
}

/*************** COMMON WORK ***********************/
/* Processes a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */

/* Processes a single pixel and returns the value of processed pixel */
int32_t apply2d_cpu(const int8_t *f, int32_t dimension, const int32_t *input, int row, int column,
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

/* calculate start postion and end position based on different block assignment */
void *calc_start_and_end_position(work *ind_work, int start, int chunk_size)
{
    int start_pos[2];
    int end_pos[2];
    start_pos[0] = start;
    start_pos[1] = 0;
    end_pos[0] = start + chunk_size; 
    end_pos[1] = 0; 
    ind_work ->start_pos[0] = start_pos[0];
    ind_work ->start_pos[1] = start_pos[1];
    ind_work ->end_pos[0] = end_pos[0];
    ind_work ->end_pos[1] = end_pos[1];
    return NULL;
}

/* Compute target value and store it into local target based on different methods */
void *compute_target_val(work * ind_work, const int32_t *original, int32_t *loc_target, int width, int height, const filter *filter) {
    for (int row = ind_work -> start_pos[0]; row < ind_work -> end_pos[0]; row++) 
    {
        for (int col = ind_work -> start_pos[1]; col < width; col++) 
        {
        loc_target[ind_work -> num_pixels] = apply2d_cpu(filter->matrix, filter->dimension, original, row, col, width, height);
        ind_work -> num_pixels++;
        }
    }
    return NULL;
}

/* Store local max and local min to a shared array */
void *store_loc_max_min(work *ind_work, int id, int max, int min) {
    ind_work -> common -> local_max_min_arr[id*2] = max;
    ind_work -> common -> local_max_min_arr[id*2+1] = min;
    return NULL;
}

/* write result stored in loc_target to global target which represent the
   result of image processing */
void *fillup_output(work *ind_work, int loc_target[], int start, int end) {
    for (int pixel = 0; pixel < ind_work -> num_pixels;pixel++) 
    {
        ind_work -> common -> output_image[ind_work ->start_pos[0] * ind_work -> common -> width + pixel] = loc_target[pixel];
    }
    return NULL;
}

void *sharding_work(void *the_work) {
  work * ind_work = (work *) the_work;
  int loc_max, loc_min, glob_max, glob_min, start, end;
  int *loc_target;
  
  // start is the start row of the block that this thread is responsible for
  start = (ind_work -> assigned) * (ind_work->id);
  end = start;
  for (int i = start; i < start + ind_work -> assigned && i < ind_work ->common->height; i++) 
  {
      end++;
  }
  loc_target = new int[ind_work -> assigned * ind_work -> common -> width];
  //end is the ending row of this thread that this thread is responsible for

  //find start and end position for each shard
  calc_start_and_end_position(ind_work, start, end-start);

  //compute target value for assigned block and store the data into local target
  compute_target_val(ind_work, ind_work ->common ->original_image, loc_target, ind_work ->common ->width, ind_work -> common ->height, ind_work ->common ->f);
  
  //find local max and local min for assigned block
  find_max_and_min(loc_target, ind_work ->num_pixels, loc_max, loc_min);

  //store local max and local min to common place
  store_loc_max_min(ind_work, ind_work -> id, loc_max, loc_min);
  
  //wait for other threads to finish their computation and store their local max and min
  pthread_barrier_wait(&ind_work -> common -> barrier);
  
  //find global max and min
  find_max_and_min(ind_work -> common -> local_max_min_arr, ind_work -> common -> max_threads * 2, glob_max, glob_min);
  
  //normalize each pixel on local target
  for (int ind = 0; ind < ind_work ->num_pixels; ind++) {
    normalize_pixel_cpu(loc_target, ind, glob_min, glob_max);
  }

  //prevent simutaneously writing and false sharing
  pthread_mutex_lock(&target_mutex);
  fillup_output(ind_work, loc_target, start, end);
  pthread_mutex_unlock(&target_mutex);
  
  //free up dynamically allocated memory and clean up thread specific resource
  delete[] loc_target;
  pthread_exit(NULL);
  return NULL;
}

void run_best_cpu(const int8_t *filter_matrix, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
    filter f;
    f.matrix = filter_matrix;
    f.dimension = dimension;

    int num_threads = 4;
    int num_threads_gene = 0;
    int assigned_row, left_row;

    assigned_row = (num_threads + height - 1)/num_threads;
    left_row = height;

    while (left_row > 0) {
      left_row-=assigned_row;
      num_threads_gene++;
    }

    common_work common;
    common.f = &f;
    common.max_threads = num_threads_gene;
    common.width = width;
    common.height = height;
    common.original_image = input;
    common.output_image = output;

    //initialize mutexs and threads and barrier
    pthread_t threads[num_threads_gene];
    pthread_mutex_init(&normalizable_mutex, NULL);
    pthread_mutex_init(&target_mutex, NULL);
    pthread_mutex_init(&debug_mutex, NULL);
    pthread_barrier_init(&common.barrier, NULL, num_threads_gene);

    //initialize work array for Sharded rows
    work works[num_threads_gene];

    common.local_max_min_arr = new int32_t[num_threads_gene * 2];
    for(int thread_id = 0; thread_id<num_threads_gene; thread_id++) {
        works[thread_id].num_pixels = 0;
        works[thread_id].common = &common;
        works[thread_id].id = thread_id;
        works[thread_id].assigned = (num_threads + height - 1)/num_threads;
        pthread_create(&threads[thread_id], NULL, sharding_work, &works[thread_id]);
    }
        
    //wait for all generated threads to join
    for (int i = 0; i < num_threads_gene; i++) 
    {
        pthread_join(threads[i], NULL);
    }
    
    //destroy every mutex, free any dynamically allocated memory
    delete[] common.local_max_min_arr;
    pthread_mutex_destroy(&normalizable_mutex);
    pthread_mutex_destroy(&target_mutex);
    pthread_mutex_destroy(&debug_mutex);
    pthread_barrier_destroy(&common.barrier);
}
