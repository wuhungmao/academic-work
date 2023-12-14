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

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "time_util.h"

//TODO: parallelize the code and optimize performance

typedef struct course{
	int grades_count;
	int average;
	char *name;
	grade_record *grades;
} course;

// Compute the historic average grade for a given course. Updates the average value in the record
void *compute_average(void *course)
{
	course_record* the_course = (course_record *) course;
	assert(the_course != NULL);
	assert(the_course->grades != NULL);
	the_course->average = 0.0;
	double local_storage = 0.0;
	for (int i = 0; i < the_course->grades_count; i++) {
		local_storage += the_course->grades[i].grade;
	}
	local_storage /= the_course->grades_count;
	the_course->average = local_storage;
	printf("course->average: %f\n", the_course->average);
	return NULL;
}

// Compute the historic average grades for all the courses
void compute_averages(course_record *courses, int courses_count)
{
	assert(courses != NULL);

	assert(courses != NULL);
	pthread_t thread[courses_count];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (int i = 0; i < courses_count; i++) {
		pthread_create(&thread[i], &attr, compute_average,  &(courses[i]));
	}
	for (int j = 0; j < courses_count; j++) {
		pthread_join(thread[j], NULL);
	}
	pthread_attr_destroy(&attr);
}


int main(int argc, char *argv[])
{
	course_record *courses;
	int courses_count;
	// Load data from file; "part2data" is the default file path if not specified
	if (load_data((argc > 1) ? argv[1] : "part2data", &courses, &courses_count) < 0) return 1;

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	compute_averages(courses, courses_count);
	clock_gettime(CLOCK_MONOTONIC, &end);

	for (int i = 0; i < courses_count; i++) {
		printf("%s: %f\n", courses[i].name, courses[i].average);
	}

	printf("%f\n", timespec_to_msec(difftimespec(end, start)));

	free_data(courses, courses_count);
	return 0;
}
