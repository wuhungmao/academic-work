// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367H5 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion, Alexey Khrabrov
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2022 Bogdan Simion
// -------------

#define _GNU_SOURCE

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include "time_util.h"
#include <string.h>
double calcute_time(int array[], int stride_size) {
    struct timespec start_time, end_time;
    //calculate array_size, by dividing amount of bytes
    // the array hold by size of an integer
    int array_size = stride_size/sizeof(1);
    //calculate next index on a list that jump across 64 bytes
    int jump = (64/sizeof(1));
    //To limit memory operation to happen within certain bound
    //this bound is array_size which represents a possible cache
    //size
    int limit = 0;
    if (clock_gettime(CLOCK_MONOTONIC, &start_time) == -1) {
        return -1;
    }
    for (int j = 0; j < 10000; j++) {
        if(limit>array_size){
            limit-=array_size;
        }
        //memory operation
        array[limit] = 5;
        limit+=jump;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end_time) == -1) {
        return -1;
    }
    return timespec_to_nsec(difftimespec(end_time, start_time));
}
double compute_cache_latency(int estimated_cache_size) {
    double total_cache_latency = 0;
    //convert kilobytes to bytes
    int stride_size = estimated_cache_size*1024;
    //create a array that is capable of holding stride_size amount of data
    int *array = (int *) malloc(stride_size);
    //warm up cache
    for (int i = stride_size/sizeof(1); i>0; i--) {
        array[i]= 2;
    }
    for (int iteration = 0; iteration < 10000; iteration++) {
        //do the experiment 10000 times
        total_cache_latency += calcute_time(array, stride_size);
    }
    return total_cache_latency;
}
void calculate_stride_size_and_mem_latency(int x_axis[], double y_axis[], int cache_sizes[], int num_of_estimated_data) {
    //assign value to each place on x and y axis
    for (int index = 0; index < num_of_estimated_data; index++) {
        x_axis[index] = cache_sizes[index];
        //compute cache latency given a specific size
        y_axis[index] += compute_cache_latency(cache_sizes[index]);
    }
}
int main(int argc, char *argv[])
{
    // how does scheduling affect accessing memory
    // Pin the thread to a single CPU to minimize the effects of scheduling
    // Don't use CPU #0 if possible, as it tends to be busier with servicing interrupts
    srandom(time(NULL));
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((random() ?: 1) % get_nprocs(), &set);
    if (sched_setaffinity(getpid(), sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        return 1;
    }
    //run part 1 a) once
    if (atoi(argv[1]) == 3) {
        //part 1 a)
        //Initializes/declares variables and the array
        int array_size = 1000000;
        char array[array_size];
        double elapsed_time = 0.0;
        double data_transfered = 0.0;
        //warm up caches
        for (int i = 999999; i>0; i--) {
            memset(&array[i], 'c', 4);
        }
        int num_of_cache_line_evicted = array_size/(64/sizeof('a'));
        //run the experiment 10 time
        struct timespec start_time, end_time;
        if (clock_gettime(CLOCK_MONOTONIC, &start_time) == -1) {
            return -1;
        }
        //Start issuing a large number of memory write operation        
        for (int j = 0; j < array_size; j+=64/sizeof('a')) {
            memset(&array[j], 'd', 4);
        }
        if (clock_gettime(CLOCK_MONOTONIC, &end_time) == -1) {
            return -1;
        } 
        elapsed_time = timespec_to_sec(difftimespec(end_time, start_time));
        //each cache line write 64 bytes to memory, (64 * num_of_cache_line_evicted) 
        //caculate total amount of bytes written to memory
        data_transfered = 64 * num_of_cache_line_evicted;
        FILE* file;
        file = fopen("Bandwidth.txt", "w");
        //Then, data_transfered is divided by elapsed_time and then convert to gigabytes
        fprintf(file, "Bandwidth: %f GB/sec", data_transfered/(elapsed_time*1000000000));
        fclose(file);
    }
    //part 1 B)
    //These three arrays contains estimated cache size for L1, L2, L3 cache
    int estimated_L1_cache_size[20] = 
    {2, 4, 8, 10, 14,\
     16, 20, 24, 28, 32,\
     36, 40, 48, 56, 64,\
     72, 81, 90, 96, 108};
    
    int estimated_L2_cache_size[20] = 
    {56, 72, 96, 128, 144,\
     160, 176, 192, 208, 224,\
     240, 256, 288, 320, 352,\
     384, 448, 512, 544, 576};
    
    int estimated_L3_cache_size[20] = 
    {10000, 11500, 13000, 14500, 16000,\
     17500, 19000, 20500, 22000, 23500,\
     25000, 26500, 28000, 29500, 31000,\
     32500, 34000, 35500, 37000, 38500};

    //x_axis represent stride size, y axis represent cache latency in nanoseconds
    int x_axis[20];
    double y_axis[20];
    //invoking ./part1 three times in "automate_collection.sh", each time collects data for a cache. 
    switch (atoi(argv[1]))
    {
    case 1:
        //collect data for L1
        calculate_stride_size_and_mem_latency(x_axis, y_axis, estimated_L1_cache_size, 20);
        break;
    case 2:
        //collect data for L2
        calculate_stride_size_and_mem_latency(x_axis, y_axis, estimated_L2_cache_size, 20);
        break;
    case 3:
        //collect data for L3
        calculate_stride_size_and_mem_latency(x_axis, y_axis, estimated_L3_cache_size, 20);
        break;
    }
    FILE* file_pointer;
    char data_file_name[100];
    //create file name for caches
    sprintf(data_file_name, "collected_data_for_L%i.dat", atoi(argv[1]));
    file_pointer = fopen(data_file_name, "w");
    for (int data = 0; data < 20; data++) {
        //print data to created file
        fprintf(file_pointer, "%d %lf %d\n", x_axis[data], y_axis[data]/100000000, x_axis[data]);
    }
    fclose(file_pointer);
}   