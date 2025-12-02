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

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "join.h"
#include "options.h"


int main(int argc, char *argv[])
{
	const char *path = parse_args(argc, argv);
	if (path == NULL) return 1;

	if (!opt_replicate && !opt_symmetric) {
		fprintf(stderr, "Invalid arguments: parallel algorithm (\'-r\' or \'-s\') is not specified\n");
		print_usage(argv);
		return 1;
	}

	if (opt_nthreads <= 0) {
		fprintf(stderr, "Invalid arguments: number of threads (\'-t\') not specified\n");
		print_usage(argv);
		return 1;
	}
	omp_set_num_threads(opt_nthreads);

	int students_count, tas_count;
	student_record *students;
	ta_record *tas;
	if (load_data(path, &students, &students_count, &tas, &tas_count) != 0) return 1;

	int result = 1;
	join_func_t *join_f = opt_nested ? join_nested : (opt_merge ? join_merge : join_hash);

	double t_start = omp_get_wtime();

	int count = 0;
	int num_stud_records_per_thread, num_ta_records_per_thread, thread_num;
	if(opt_replicate)
	{
		//fragment and replicate
		if (students_count > tas_count) 
		{
			//partition students record and copy tas records
			num_stud_records_per_thread = (students_count + opt_nthreads-1)/opt_nthreads;
			#pragma omp parallel reduction(+:count) firstprivate(tas) private(thread_num) shared(num_stud_records_per_thread, tas_count, students, opt_nthreads) 
			{
				student_record *stud_records = (student_record *) malloc(sizeof(student_record) * num_stud_records_per_thread);
				thread_num = omp_get_thread_num();
				if ((thread_num+1) == opt_nthreads) 
				{
					//process remaining student records in last thread
					int last_thread_num_stud_records = 0;
					for(int i = 0; i < num_stud_records_per_thread; i++) 
					{
						if (i+thread_num*num_stud_records_per_thread < students_count)
						{
							stud_records[i] = students[i + thread_num*num_stud_records_per_thread];
							last_thread_num_stud_records++;
						}
					}
					count = join_f(stud_records, last_thread_num_stud_records, tas, tas_count);
				}
				else 
				{
					//process a fixed number of student records in every thread except last thread
					for(int i = 0; i < num_stud_records_per_thread; i++) 
					{
						stud_records[i] = students[i + thread_num*num_stud_records_per_thread];
					}
					count = join_f(stud_records, num_stud_records_per_thread, tas, tas_count);
				}
				free(stud_records);
			}
		} 
		else 
		{
			//partition tas record and copy student records
			num_ta_records_per_thread = (tas_count + opt_nthreads-1)/opt_nthreads;
			int last_thread_num_ta_records = 0;
			#pragma omp parallel reduction(+: count) firstprivate(students, last_thread_num_ta_records) private(thread_num) shared(num_ta_records_per_thread, students_count, tas, opt_nthreads)
			{
				ta_record *ta_records = (ta_record *) malloc(sizeof(ta_record) * num_ta_records_per_thread);
				thread_num = omp_get_thread_num();
				if ((thread_num+1) == opt_nthreads) 
				{
					//process remaining ta records in last thread
					for(int i = 0; i < num_ta_records_per_thread; i++) 
					{
						if (i+thread_num*num_ta_records_per_thread < tas_count)
						{
							ta_records[i] = tas[i + thread_num*num_ta_records_per_thread];
							last_thread_num_ta_records++;
						} else {
							break;
						}
					}
					count = join_f(students, students_count, ta_records, last_thread_num_ta_records);
				} 
				else 
				{
					//process a fixed number of ta records in every thread except last thread
					for(int j = 0; j < num_ta_records_per_thread; j++) 
					{
						ta_records[j] = tas[j + thread_num*num_ta_records_per_thread];
					}
					count = join_f(students, students_count, ta_records, num_ta_records_per_thread);
				}
				free(ta_records);
			}
		}
	}
	else
	{
		//symmetric partitioning
		num_stud_records_per_thread = (students_count)/opt_nthreads;
		int stud_ids[opt_nthreads];
		int stud_ids_ind[opt_nthreads];
		int lower_bound, upper_bound, start_Ind, end_ind;

		for(int i = 0; i < opt_nthreads-1; i++)
		{
			stud_ids[i] = students[(i+1)*num_stud_records_per_thread-1].sid;
			stud_ids_ind[i] = ((i+1)*num_stud_records_per_thread)-1;
		}

		stud_ids[opt_nthreads-1] = students[students_count-1].sid;
		stud_ids_ind[opt_nthreads-1] = students_count-1;

		#pragma omp parallel reduction(+:count) private(thread_num, lower_bound, upper_bound, start_Ind, end_ind) shared(num_stud_records_per_thread, students, tas, tas_count, stud_ids) 
		{
			student_record *student_records = (student_record *) malloc(sizeof(student_record) * 2 * num_stud_records_per_thread);
			ta_record *ta_records = (ta_record *) malloc(sizeof(ta_record) * tas_count);

			thread_num = omp_get_thread_num();
			if(thread_num == 0) 
			{
				lower_bound = -1;
				upper_bound = stud_ids[0];
				start_Ind = 0;
				end_ind = stud_ids_ind[0];
			}
			else
			{
				lower_bound = stud_ids[thread_num-1];
				upper_bound = stud_ids[thread_num];
				start_Ind = stud_ids_ind[thread_num-1]+1;
				end_ind = stud_ids_ind[thread_num];
			}
			
			//Find the part of student records that is responsible by this thread
			int num_stud_records_in_this_thread = 0;
			for(int i = start_Ind; i <= end_ind; i++) 
			{
				student_records[num_stud_records_in_this_thread] = students[i];
				num_stud_records_in_this_thread++;
			}

			//Find the part of ta records that is reponsible by this thread
			int num_ta_records_in_this_thread;
		
			num_ta_records_in_this_thread = 0;
			for(int j = 0; j < tas_count; j++)
			{
				if((tas[j].sid>lower_bound) && (tas[j].sid<=upper_bound))
				{
					ta_records[num_ta_records_in_this_thread] = tas[j];
					num_ta_records_in_this_thread++;
				}
			}

			count = join_f(student_records, num_stud_records_in_this_thread, ta_records, num_ta_records_in_this_thread);
			free(student_records);
			free(ta_records);
		}
	}
	double t_end = omp_get_wtime();

	if (count < 0) goto end;
	printf("%d\n", count);
	printf("%f\n", (t_end - t_start) * 1000.0);
	result = 0;

end:
	free(students);
	free(tas);
	return result;
}
