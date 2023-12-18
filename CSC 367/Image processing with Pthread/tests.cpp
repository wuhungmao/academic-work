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

#include "filters.h"
#include "pgm.h"
#include "gtest/gtest.h"

TEST(filters, identity_filter_is_applied_to_simple_matrix) {
  constexpr int image_size = 9;
  int32_t image[image_size] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int32_t> target(image_size);
  apply_filter2d(builtin_filters[IDENTITY_FILTER], image, target.data(), 3, 3);

  for (int i = 0; i < image_size; i++) {
    ASSERT_EQ(image[i], target[i]) << i;
  }
}

void test_all_threads_and_chunks(int width, int height, const int32_t *image,
                                 const int32_t *expected, filter *f) 
                                 {

  for (int num_threads = 2; num_threads < 3; num_threads++) {
    /*
    {
      std::vector<int32_t> target(width * height);
      apply_filter2d(f, image, target.data(), width, height);

      for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(target[i], expected[i]) << i;
      }
      printf("sequential part finished for %i num of threads\n", num_threads);
    }
    {
      std::vector<int32_t> target(width * height);
      apply_filter2d_threaded(f, image, target.data(), width, height,
                              num_threads, parallel_method::SHARDED_ROWS, 0);

      for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(target[i], expected[i]) << i;
      }
      printf("sharded rows finished for %i num of thread\n", num_threads);
    }
    {
      std::vector<int32_t> target(width * height);
      apply_filter2d_threaded(f, image, target.data(), width, height,
                              num_threads,
                              parallel_method::SHARDED_COLUMNS_COLUMN_MAJOR, 0);

      for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(target[i], expected[i]) << i;
      }
      printf("sharded columns column major finished for %i num of thread\n", num_threads);
    }
    {
      std::vector<int32_t> target(width * height);
      apply_filter2d_threaded(f, image, target.data(), width, height,
                              num_threads,
                              parallel_method::SHARDED_COLUMNS_ROW_MAJOR, 0);

      for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(target[i], expected[i]) << i;
      }
      printf("sharded columns row major finished for %i num of thread\n", num_threads);
    }
    */
    int target_0;
    for (int chunk_size = 1; chunk_size < 18; chunk_size++) {
      std::vector<int32_t> target(width * height);
      apply_filter2d_threaded(f, image, target.data(), width, height,
                              num_threads, parallel_method::WORK_QUEUE,
                              chunk_size);

      for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(target[i], expected[i]) << i;
      }
    }
  }
                                 }
  
                                 


TEST(filters, laplacian_filter_produces_correct_result_5_3) {
  constexpr int width = 5;
  constexpr int height = 5;
  constexpr int32_t image[] = {1, 4, 7, 10, 3, 2, 5, 8, 1, 4, 3, 6, 9, 2, 5, 6, 8, 7, 4, 3, 6, 3, 2, 1, 5};
  constexpr int32_t expected[] = {161, 135, 119, 0,  161,
                                  156, 150, 98, 255, 114,
                                  161, 156, 83,  208, 93,
                                  114, 98, 124, 135, 161,
                                  72, 171, 166, 187, 67};
  test_all_threads_and_chunks(width, height, image, expected,
                              builtin_filters[LAPLACIAN_FILTER_3]);
}

TEST(filters, laplacian_filter_produces_correct_result_1_1) {
  constexpr int width = 1;
  constexpr int height = 1;
  constexpr int32_t image[] = {17};
  constexpr int32_t expected[] = {-68};
  test_all_threads_and_chunks(width, height, image, expected,
                              builtin_filters[LAPLACIAN_FILTER_3]);
}

TEST(filters, laplacian_filter_produces_correct_result_2_1) {
  constexpr int width = 2;
  constexpr int height = 1;
  constexpr int32_t image[] = {10, 5};
  constexpr int32_t expected[] = {0, 255};
  test_all_threads_and_chunks(width, height, image, expected,
                              builtin_filters[LAPLACIAN_FILTER_3]);
}

// TODO: add more tests here. Some suggestions:
// Test a 2x2 image, a prime number dimension image, e.g. 23 x 11.
// Tests all of the above with different filters.
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
