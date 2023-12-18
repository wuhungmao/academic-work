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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "kernels.h"
#include "pgm.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out) {
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                time_gpu_transfer_out));
}

int main(int argc, char **argv) {
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3) {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }

  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
    case 'i':
      input_filename = std::string(optarg);
      break;
    case 'o':
      cpu_output_filename = std::string(optarg);
      base_gpu_output_filename = std::string(optarg);
      break;
    default:
      return 0;
    }
  }

  pgm_image source_img;
  init_pgm_image(&source_img);

  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
    printf("Error loading source image.\n");
    return 0;
  }

  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
         "Speedup_noTrf Speedup\n");

  // Create a filter and filter dimension
  const int8_t FILTER[] = 
  {
    0, 1, 0,
    1, -4, 1,
    0, 1, 0,
  };
  const int FILTER_DIMENSION = 3;

  /* TODO: run your CPU implementation here and get its time. Don't include
   * file IO in your measurement. Store the time taken in a variable, so
   * it can be printed later for comparison with GPU kernels. */
  /* For example: */
  
  //why do you put it in a block
  std::string cpu_file = cpu_output_filename;
  pgm_image cpu_output_img;
  copy_pgm_image_size(&source_img, &cpu_output_img);
  // Timing cpu runtime

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float cpu_time = 0.0;
  cudaEventRecord(start);

  run_best_cpu(FILTER, FILTER_DIMENSION, source_img.matrix,
                cpu_output_img.matrix, source_img.width, source_img.height);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time, start, stop);
      
  save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
  destroy_pgm_image(&cpu_output_img);

  /* TODO:
   * run each of your gpu implementations here,
   * get their time,
   * and save the output image to a file.
   * Don't forget to add the number of the kernel
   * as a prefix to the output filename:
   * Print the execution times by calling print_run().
   */

  /* For example: */

  float transfer_in_time;
  float computation_time;
  float transfer_out_time;

  //kernel 1
  std::string gpu_file = "1" + base_gpu_output_filename;
  pgm_image gpu_output_img_1;
  copy_pgm_image_size(&source_img, &gpu_output_img_1);

  run_kernel1(FILTER, FILTER_DIMENSION, source_img.matrix,
                gpu_output_img_1.matrix, source_img.width, source_img.height, &transfer_in_time, &computation_time, &transfer_out_time);  // From kernels.h
  print_run(cpu_time, 1, computation_time, transfer_in_time, transfer_out_time);    // Defined on the top of this file

  save_pgm_to_file(gpu_file.c_str(), &gpu_output_img_1);
  destroy_pgm_image(&gpu_output_img_1);

  //kernel 2
  gpu_file = "2" + base_gpu_output_filename;
  pgm_image gpu_output_img_2;
  copy_pgm_image_size(&source_img, &gpu_output_img_2);

  run_kernel2(FILTER, FILTER_DIMENSION, source_img.matrix,
                gpu_output_img_2.matrix, source_img.width, source_img.height, &transfer_in_time, &computation_time, &transfer_out_time);  // From kernels.h
  print_run(cpu_time, 2, computation_time, transfer_in_time, transfer_out_time);    // Defined on the top of this file

  save_pgm_to_file(gpu_file.c_str(), &gpu_output_img_2);
  destroy_pgm_image(&gpu_output_img_2);

  //kernel 3
  gpu_file = "3" + base_gpu_output_filename;
  pgm_image gpu_output_img_3;
  copy_pgm_image_size(&source_img, &gpu_output_img_3);

  run_kernel3(FILTER, FILTER_DIMENSION, source_img.matrix,
                gpu_output_img_3.matrix, source_img.width, source_img.height, &transfer_in_time, &computation_time, &transfer_out_time);  // From kernels.h
  print_run(cpu_time, 3, computation_time, transfer_in_time, transfer_out_time);    // Defined on the top of this file

  save_pgm_to_file(gpu_file.c_str(), &gpu_output_img_3);
  destroy_pgm_image(&gpu_output_img_3);

  //kernel 4
  gpu_file = "4" + base_gpu_output_filename;
  pgm_image gpu_output_img_4;
  copy_pgm_image_size(&source_img, &gpu_output_img_4);

  run_kernel4(FILTER, FILTER_DIMENSION, source_img.matrix,
                gpu_output_img_4.matrix, source_img.width, source_img.height, &transfer_in_time, &computation_time, &transfer_out_time);  // From kernels.h
  print_run(cpu_time, 4, computation_time, transfer_in_time, transfer_out_time);    // Defined on the top of this file

  save_pgm_to_file(gpu_file.c_str(), &gpu_output_img_4);
  destroy_pgm_image(&gpu_output_img_4);

  // kernel 5
  gpu_file = "5" + base_gpu_output_filename;
  pgm_image gpu_output_img_5;
  copy_pgm_image_size(&source_img, &gpu_output_img_5);

  run_kernel5(FILTER, FILTER_DIMENSION, source_img.matrix,
                gpu_output_img_5.matrix, source_img.width, source_img.height, &transfer_in_time, &computation_time, &transfer_out_time);  // From kernels.h
  print_run(cpu_time, 5, computation_time, transfer_in_time, transfer_out_time);    // Defined on the top of this file

  save_pgm_to_file(gpu_file.c_str(), &gpu_output_img_5);
  destroy_pgm_image(&gpu_output_img_5);
}
