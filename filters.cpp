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

//Q3:should i modify setting file? not sure how to solve it
#include <queue>
#include "filters.h"
#include <pthread.h>
#include <iostream>
#include <thread>

enum work_types {
  COMPUTE_LOCAL_TARGET = 0,
  NORMALIZE_LOCAL_TARGET = 1
};

typedef struct work_t work;

struct common_work_t; 

typedef struct work_t
{
  common_work_t *common;
  int32_t id;
  int32_t assigned;
  int32_t num_pixels;
  int32_t start_pos[2];
  int32_t end_pos[2];
  int32_t method;
  work_types work_type;
} work;

typedef struct common_work_t
{
  const filter *f;
  const int32_t *original_image;
  int32_t *output_image;
  int32_t width;
  int32_t height;
  int32_t max_threads;
  int32_t total_work;
  int32_t* local_max_min_arr;
  work* work_queue;
  std::queue<int> work_pool;
  pthread_barrier_t barrier;
  pthread_cond_t normalizable;
  int32_t normalizable_cond;
} common_work;

pthread_mutex_t normalizable_mutex;
pthread_mutex_t target_mutex;
pthread_mutex_t work_pool_mutex;
pthread_mutex_t debug_mutex;

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] = {
    0, 1, 0, 1, -4, 1, 0, 1, 0,
};
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] = {
    0, 1, 1, 2, 2, 2,   1,   1,   0, 1, 2, 4, 5, 5,   5,   4,   2,
    1, 1, 4, 5, 3, 0,   3,   5,   4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3,   -12, -24, -12,
    3, 5, 2, 1, 4, 5,   3,   0,   3, 5, 4, 1, 1, 2,   4,   5,   5,
    5, 4, 2, 1, 0, 1,   1,   2,   2, 2, 1, 1, 0,
};
filter log_f = {9, log_m};

/* Identity filter */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) 
{
  if (smallest == largest) {
    return;
  }
  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/* find out all locals for every pixel assigned to a block */
void find_pixel_pos(int (&arr)[][2], int dim, int col, int row, int width, int height) {
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
void find_max_and_min(int target[], int size, int& max, int& min) {
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
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
                int32_t width, int32_t height, int row, int column) {
                int val = 0;
                int array[f->dimension*f->dimension][2];
                switch (f->dimension)
                {
                case 1:
                  //identity filter
                  val = original[width * row + column] * f->matrix[0];
                  break;
                case 3:
                  // 3 dimensional filter
                  find_pixel_pos(array, f -> dimension, column, row, width, height);
                  for (int i = 0; i < 9; i++) {
                    if (array[i][0] != -1) {
                      val += f -> matrix[i] * original[array[i][0] * width + array[i][1]];
                    }
                  }
                  break;
                case 5:
                  // 5 dimensional filter
                  find_pixel_pos(array, f -> dimension, column, row, width, height);
                  for (int i = 0; i < 25; i++) {
                    if (array[i][0] != -1) {
                      val += f -> matrix[i] * original[array[i][0] * width + array[i][1]];
                    }
                  }
                  break;
                case 9:
                  // 9 dimensional filter
                  find_pixel_pos(array, f -> dimension, column, row, width, height);
                  for (int i = 0; i < 81; i++) {
                    if (array[i][0] != -1) {
                      val += f -> matrix[i] * original[array[i][0] * width + array[i][1]];
                    }
                  }
                  break;
                }
                return val;
  return 0;
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/
void apply_filter2d( const filter *f, const int32_t *original, int32_t *target,
                    int32_t width, int32_t height) {
                      for (int row = 0; row < height; row++) {
                        for (int col = 0; col < width; col++) {
                          target[col + row * width] = apply2d(f, original, target, width, height, row, col);
                        }
                      }
                      int max = 0;
                      int min = 0;
                      int size = width*height;
                      find_max_and_min(target, size, max, min);
                      for (int i = 0; i < size; i++) {
                        normalize_pixel(target, i, min, max);
                      }
                    }

/* calculate start postion and end position based on different block assignment */
void *calc_start_and_end_position(work *ind_work, int start, int chunk_size)
{
  int start_pos[2];
  int end_pos[2];
  switch (ind_work->method) {
    case parallel_method::SHARDED_ROWS:
      start_pos[0] = start;
      start_pos[1] = 0;
      end_pos[0] = start + chunk_size; 
      end_pos[1] = 0; 
      break;
    case parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR:
      start_pos[0] = 0;                 
      start_pos[1] = start;
      end_pos[0] = 0;                   
      end_pos[1] = start + chunk_size;  
      break;
    case parallel_method::SHARDED_COLUMNS_ROW_MAJOR:
      start_pos[0] = 0;                 
      start_pos[1] = start;
      end_pos[0] = 0;                   
      end_pos[1] = start + chunk_size;
      break;
  }
  ind_work ->start_pos[0] = start_pos[0];
  ind_work ->start_pos[1] = start_pos[1];
  ind_work ->end_pos[0] = end_pos[0];
  ind_work ->end_pos[1] = end_pos[1];
  return NULL;
}

/* Compute target value and store it into local target based on different methods */
void *compute_target_val(work * ind_work, const int32_t *original, int32_t *loc_target, int width, int height, const filter *filter) {
    switch(ind_work -> method){
      case parallel_method::SHARDED_ROWS:
        for (int row = ind_work -> start_pos[0]; row < ind_work -> end_pos[0]; row++) {
          for (int col = ind_work -> start_pos[1]; col < width; col++) {
              loc_target[ind_work -> num_pixels] = apply2d(filter, original, loc_target, width, height, row, col);
              ind_work -> num_pixels++;
          }
        }
      break;
      case parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR:
        for (int col = ind_work -> start_pos[1]; col < ind_work -> end_pos[1]; col++) {
          for (int row = ind_work -> start_pos[0]; row < height; row++) {
              loc_target[ind_work -> num_pixels] = apply2d(filter, original, loc_target, width, height, row, col);
              ind_work -> num_pixels++;
          }
        }
      break;
      case parallel_method::SHARDED_COLUMNS_ROW_MAJOR:
      for (int row = ind_work -> start_pos[0]; row < height; row++) {
        for (int col = ind_work -> start_pos[1]; col < ind_work -> end_pos[1]; col++) {
              loc_target[ind_work -> num_pixels] = apply2d(filter, original, loc_target, width, height, row, col);
              ind_work -> num_pixels++;
          }
        }
      break;
      case parallel_method::WORK_QUEUE:
        for (int row = ind_work -> start_pos[0]; row < ind_work -> end_pos[0]+1 && row < ind_work ->common ->height; row++) {
          for (int col = ind_work -> start_pos[1]; col < ind_work -> end_pos[1]+1 && col < ind_work -> common -> width; col++) {
            loc_target[ind_work -> num_pixels] = apply2d(filter, original, loc_target, width, height, row, col);
            ind_work -> num_pixels++;
          }
        }
      break;
    }
  return NULL;
}

/* Store local max and local min to a shared array */
void *store_loc_max_min(work *ind_work, int id, int max, int min) {
  ind_work -> common -> local_max_min_arr[id*2] = max;
  ind_work -> common -> local_max_min_arr[id*2+1] = min;
  return NULL;
}

/* This method is specific to work queue implementation. It initializes a new work
   which shares same information as the old work except for work type. Work type 
   is changed to NORMALIZE_LOCAL_TARGET. */
void *change_work_type(work *ind_work, work &new_work) {
  new_work.work_type = NORMALIZE_LOCAL_TARGET;
  new_work.common = ind_work -> common;
  new_work.id = ind_work -> id + ind_work ->common -> total_work;
  new_work.num_pixels = ind_work -> num_pixels;
  new_work.method = ind_work->method;
  new_work.start_pos[0] = ind_work -> start_pos[0];
  new_work.start_pos[1] = ind_work -> start_pos[1];
  new_work.end_pos[0] = ind_work -> end_pos[0];
  new_work.end_pos[1] = ind_work -> end_pos[1];
  return NULL;
}

/* write result stored in loc_target to global target which represent the
   result of image processing */
void *fillup_output(work *ind_work, int loc_target[], int start, int end) {
    int ind_output = start;
    int ind_target = 0;
    switch (ind_work -> method)
    {
    case parallel_method::SHARDED_ROWS:
      for (int pixel = 0; pixel < ind_work -> num_pixels;pixel++) {
        ind_work -> common -> output_image[ind_work ->start_pos[0] * ind_work -> common -> width + pixel] = loc_target[pixel];
      }
      break;
    case parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR:
      for(int i = 0; i< end - start; i++) {
        ind_output = start+i;
        for (int row = 0; row < ind_work -> common -> height; row++) {
          ind_work -> common -> output_image[ind_output] = loc_target[ind_target];
          ind_output+=ind_work->common->width;
          ind_target++;
        }
      }
      break;
    case parallel_method::SHARDED_COLUMNS_ROW_MAJOR:
      for (int row = 0; row < ind_work -> common -> height; row++) {
        for(int col = 0; col < end - start; col++){
          ind_work -> common -> output_image[ind_output + col] = loc_target[ind_target];
          ind_target++;
        }
        ind_output+=ind_work->common->width;
      }
      break;
    case parallel_method::WORK_QUEUE:
      int target_ind = 0;
        for (int row = ind_work -> start_pos[0]; row < ind_work -> end_pos[0]+1 && row < ind_work ->common ->height; row++) {
          for (int col = ind_work -> start_pos[1]; col < ind_work -> end_pos[1]+1 && col < ind_work -> common -> width; col++) {
            ind_work -> common ->output_image[row * ind_work -> common ->width + col] = loc_target[target_ind];
            target_ind++;
          }
        }
    }
  return NULL;
}

/****************** ROW/COLUMN SHARDING ************/
/* This method does the work for Sharded_rows, sharded columns column major and sharded columns row major methods. */
void *sharding_work(void *the_work) {
  work * ind_work = (work *) the_work;
  int loc_max, loc_min, glob_max, glob_min, start, end;
  int *loc_target;
  
  switch (ind_work -> method)
  {
  case parallel_method::SHARDED_ROWS:
    // start is the start row of the block that this thread is responsible for
    start = (ind_work -> assigned) * (ind_work->id);
    end = start;
    for (int i = start; i < start + ind_work -> assigned && i < ind_work ->common->height; i++) 
    {
      end++;
    }
    loc_target = new int[ind_work -> assigned * ind_work -> common -> width];
    //end is the ending row of this thread that this thread is responsible for
    break;
  case parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR:
    // start is the start column of the block that this thread is responsible for
    start = (ind_work -> assigned) * (ind_work->id);
    end = start;
    for (int i = start; i < start + ind_work -> assigned && i < ind_work ->common->width; i++) 
    {
      end++;
    }
    loc_target = new int[ind_work -> assigned * ind_work -> common -> height];
    //end is the ending column of this thread that this thread is responsible for
    break;
  case parallel_method::SHARDED_COLUMNS_ROW_MAJOR:
    // start is the start column of the block that this thread is responsible for
    start = (ind_work -> assigned) * (ind_work->id);
    end = start;
    for (int i = start; i < start + ind_work -> assigned && i < ind_work ->common->width; i++) 
    {
      end++;
    }
    loc_target = new int[ind_work -> assigned * ind_work -> common -> height];
    //end is the ending column of this thread that this thread is responsible for
    break;
  }

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
    normalize_pixel(loc_target, ind, glob_min, glob_max);
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

/* initialize each indivial work based on assigned work chunk and store them
   into work pool */
void *initialize_work_pool(common_work &common, int width, int height, int work_chunk) {
    int work_num = 0;
    for(int i = 0; i<height; i+=work_chunk) {
      for (int j = 0; j<width; j+=work_chunk) {
        common.work_pool.push(work_num);
        work_num++;
      }
    }
    common.local_max_min_arr = new int32_t[work_num * 2];
    common.total_work = work_num;
    common.work_queue = new work[work_num*2];
    int work_id = 0;
    for(int i = 0; i<height; i+=work_chunk) {
      for (int j = 0; j<width; j+=work_chunk) {
        work ind_work;
        ind_work.num_pixels = 0;
        ind_work.start_pos[0] = i;
        ind_work.start_pos[1] = j;
        ind_work.end_pos[0] = i+work_chunk-1;
        ind_work.end_pos[1] = j+work_chunk-1;
        ind_work.method = parallel_method::WORK_QUEUE;
        ind_work.id = work_id;
        ind_work.common = &common;
        ind_work.work_type = COMPUTE_LOCAL_TARGET;
        common.work_queue[work_id] = ind_work;
        work_id++;
      }
    }
    return NULL;
}

/* This method is only for work pool implementation. It is called when there are more threads than amount of work in a pool. A thread
   is assigned to this method so that pthread_join know how many threads it is waiting for in compile time */
void *do_nothing(void *nothing) {
  pthread_exit(NULL);
  return NULL;
}

/* This method is responsible for doing the work in work pool. This method execute different tasks based on a 
   work's work type. A compute_local_taregt work compute value for local target while a normalize local target
   normalize every pixel value stored within the local target. Before this method is called, all 
   compute_local_target works are determined and inserted into work queue. This method then retrieved work in 
   inserting order. If a normalize_local_target work is retrieved before all compute_local_target work compute
   its local maximum and local minimum, the normalize_local_target work block. A condition variable is used 
   to simulate the idea of a barrier in shard_work method. Once all compute_local_target compute local maximum
   and local minimum of local target, then last work would broadcast and all threads waiting would wake up and
   continue to normalize local larget. */
void *queue_work(void *the_work) { 
  work* curr_work = (work *) the_work;
  int loc_max, loc_min, glob_max, glob_min;
  int loc_target[(curr_work -> end_pos[0] - curr_work -> start_pos[0]+1) * (curr_work -> end_pos[1] - curr_work -> start_pos[1] + 1)] = {0};

  //local target will store block from left to right, up to down
  while(true) {
    switch (curr_work->work_type)
    {
    case COMPUTE_LOCAL_TARGET:
      compute_target_val(curr_work, curr_work -> common -> original_image, loc_target, curr_work -> common -> width, curr_work -> common -> height, curr_work ->common ->f);
      find_max_and_min(loc_target, curr_work -> num_pixels, loc_max, loc_min);
      store_loc_max_min(curr_work, curr_work -> id, loc_max, loc_min);
      
      //This function stores unnormalized value into array of output image. This way allow normalize_local_target work to retrieve 
      //needed data quickly since compute_local_target work and normalized_local_target share same starting position on the array. 
      pthread_mutex_lock(&target_mutex);
      fillup_output(curr_work, loc_target, 0, 0);
      pthread_mutex_unlock(&target_mutex);

      work new_work;
      change_work_type(curr_work, new_work);

      //initialize new work and insert it at the end of work queue
      pthread_mutex_lock(&work_pool_mutex);
      curr_work-> common -> work_queue[new_work.id] = new_work;
      curr_work-> common -> work_pool.push(new_work.id);
      pthread_mutex_unlock(&work_pool_mutex);

      pthread_mutex_lock(&normalizable_mutex);
      if (curr_work->id == curr_work -> common -> total_work-1){
        curr_work -> common -> normalizable_cond = 1;
        pthread_cond_broadcast(&curr_work -> common ->normalizable);
        //change condition variable and signal all waiting threads doing normalize_local_target_work;
      }
      pthread_mutex_unlock(&normalizable_mutex);
      break;
      
    case NORMALIZE_LOCAL_TARGET:
      //normalize_local_target work needs to wait until all compute_local_target work compute its local maximum and local minimum
      pthread_mutex_lock(&normalizable_mutex);
      while(curr_work -> common -> normalizable_cond == 0) {
        pthread_cond_wait(&curr_work -> common -> normalizable, &normalizable_mutex);
      }
      pthread_mutex_unlock(&normalizable_mutex);
      
      find_max_and_min(curr_work -> common -> local_max_min_arr, curr_work -> common -> total_work * 2, glob_max, glob_min);
      int target_ind = 0;
      for (int row = curr_work -> start_pos[0]; row < curr_work -> end_pos[0]+1 && row < curr_work ->common ->height; row++) {
        for (int col = curr_work -> start_pos[1]; col < curr_work -> end_pos[1]+1 && col < curr_work -> common -> width; col++) {
          loc_target[target_ind] = curr_work -> common -> output_image[row * curr_work -> common -> width + col];
          target_ind++;
        }
      }

      for (int ind = 0; ind < curr_work -> num_pixels; ind++) {
        normalize_pixel(loc_target, ind, glob_min, glob_max);
      }

      pthread_mutex_lock(&target_mutex);
      fillup_output(curr_work, loc_target, 0, 0);
      pthread_mutex_unlock(&target_mutex);
      break;
    }

    pthread_mutex_lock(&work_pool_mutex);
    int curr_ind;
    //if work_pool is empty, thread exit. Otherwise, retrieve one work from work queue
    if (curr_work -> common -> work_pool.empty()) 
    { 
      pthread_mutex_unlock(&work_pool_mutex);
      pthread_exit(NULL);
    }
    else
    {
      curr_ind = curr_work -> common -> work_pool.front();
      curr_work -> common -> work_pool.pop();
      curr_work = &curr_work -> common -> work_queue[curr_ind];
      pthread_mutex_unlock(&work_pool_mutex);
    }
  }
  return NULL;
}

/***************** MULTITHREADED ENTRY POINT ******/
/* TODO: this is where you should implement the multithreaded version
 * of the code. Use this function to identify which method is being used
 * and then call some other function that implements it.
 */
void apply_filter2d_threaded(const filter *f, const int32_t *original,
                             int32_t *target, int32_t width, int32_t height,
                             int32_t num_threads, parallel_method method,
                             int32_t work_chunk) {
  /* You probably want to define a struct to be passed as work for the
   * threads.
   * Some values are used by all threads, while others (like thread id)
   * are exclusive to a given thread.
   *
   * An uglier (but simpler) solution is to define the shared variables
   * as global variables.
   */
  //initialize common and its properties
  int num_threads_gene = 0;
  int assigned_col, assigned_row, left_row, left_col;
  switch (method)
  {
  case parallel_method::SHARDED_ROWS:
    assigned_row = (num_threads + height - 1)/num_threads;
    left_row = height;
    while (left_row >0) {
      left_row-=assigned_row;
      num_threads_gene++;
    }
    break;
  case parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR:
    assigned_col = (num_threads + width - 1)/num_threads;
    left_col = width;
    while (left_col >0) {
      left_col-=assigned_col;
      num_threads_gene++;
    }
    break;
  case parallel_method::SHARDED_COLUMNS_ROW_MAJOR:
    assigned_col = (num_threads + width - 1)/num_threads;
    left_col = width;
    while (left_col >0) {
      left_col-=assigned_col;
      num_threads_gene++;
    }
    break;
  case parallel_method::WORK_QUEUE:
    num_threads_gene = num_threads;
  }
  
  //set up common. 
  common_work common;
  common.f = f;
  common.max_threads = num_threads_gene;
  common.width = width;
  common.height = height;
  common.original_image = original;
  common.output_image = target;
  common.normalizable_cond = 0;

  //initialize mutexs and threads and barrier
  pthread_t threads[num_threads_gene];
  pthread_mutex_init(&normalizable_mutex, NULL);
  pthread_mutex_init(&target_mutex, NULL);
  pthread_mutex_init(&work_pool_mutex, NULL);
  pthread_mutex_init(&debug_mutex, NULL);
  pthread_cond_init(&common.normalizable, NULL);
  pthread_barrier_init(&common.barrier, NULL, num_threads_gene);

  //initialize work array for Sharded rows, sharded_columns_column_major,
  //sharded_columns_row_major
  work works[num_threads_gene];

  //initialize works based on thread_id and assign one work for one thread
  switch (method)
  {
  case parallel_method::SHARDED_ROWS:
    common.local_max_min_arr = new int32_t[num_threads_gene * 2];
    for(int thread_id = 0; thread_id<num_threads_gene; thread_id++) {
      works[thread_id].num_pixels = 0;
      works[thread_id].common = &common;
      works[thread_id].id = thread_id;
      works[thread_id].assigned = (num_threads + height - 1)/num_threads;
      works[thread_id].method = method;
      pthread_create(&threads[thread_id], NULL, sharding_work, &works[thread_id]);
    }
    break;
  case parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR:
    common.local_max_min_arr = new int32_t[num_threads_gene * 2];
    for(int thread_id = 0; thread_id<num_threads_gene; thread_id++) {
      works[thread_id].num_pixels = 0;
      works[thread_id].common = &common;
      works[thread_id].id = thread_id;
      works[thread_id].assigned = (num_threads + width - 1)/num_threads;
      works[thread_id].method = method;
      pthread_create(&threads[thread_id], NULL, sharding_work, &works[thread_id]);
    }
    break;
  case parallel_method::SHARDED_COLUMNS_ROW_MAJOR:
    common.local_max_min_arr = new int32_t[num_threads_gene * 2];
    for(int thread_id = 0; thread_id<num_threads_gene; thread_id++) {
      works[thread_id].num_pixels = 0;
      works[thread_id].common = &common;
      works[thread_id].id = thread_id;
      works[thread_id].assigned = (num_threads + width - 1)/num_threads;
      works[thread_id].method = method;
      pthread_create(&threads[thread_id], NULL, sharding_work, &works[thread_id]);
    }
    break;
  case parallel_method::WORK_QUEUE:
    pthread_mutex_lock(&work_pool_mutex);
    initialize_work_pool(common, width, height, work_chunk);
    int work_id;
    for (int thread_id = 0; thread_id < num_threads; thread_id++) {
      if (!common.work_pool.empty()) {
        work_id = common.work_pool.front();
        common.work_pool.pop();
        pthread_create(&threads[thread_id], NULL, queue_work, &common.work_queue[work_id]);
      }
      else {
        pthread_create(&threads[thread_id], NULL, do_nothing, NULL);
      }
    }
    pthread_mutex_unlock(&work_pool_mutex);
    break;
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
  pthread_mutex_destroy(&work_pool_mutex);
  pthread_mutex_destroy(&debug_mutex);
  pthread_cond_destroy(&common.normalizable);
  pthread_barrier_destroy(&common.barrier);
}
