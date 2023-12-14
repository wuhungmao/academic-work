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

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "join.h"
#include "options.h"


int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int result = 1;
	student_record *students = NULL;
	ta_record *tas = NULL;

	const char *path = parse_args(argc, argv);
	if (path == NULL) goto end;

	if (!opt_replicate && !opt_symmetric) {
		fprintf(stderr, "Invalid arguments: parallel algorithm (\'-r\' or \'-s\') is not specified\n");
		print_usage(argv);
		goto end;
	}

	int parts, id;
	MPI_Comm_size(MPI_COMM_WORLD, &parts);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	// Load this process's partition of data
	char part_path[PATH_MAX] = "";
	snprintf(part_path, sizeof(part_path), "%s_%d", path, id);
	int students_count, tas_count;
	if (load_data(part_path, &students, &students_count, &tas, &tas_count) != 0) goto end;

	join_func_t *join_f = opt_nested ? join_nested : (opt_merge ? join_merge : join_hash);
	int  count = -1;

	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();

	if(opt_replicate) 
	{
		//Fragment and replicate
		/* Calculate the total number of ta records*/
		int ta_num_sum, stud_num_sum, status;
		status = MPI_Allreduce(&tas_count, &ta_num_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if(status != MPI_SUCCESS)
		{
			fprintf(stderr, "MPI_Allreduce failed\n");
		}
		
		/* Calculate total number of student records*/
		status = MPI_Allreduce(&students_count, &stud_num_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if(status != MPI_SUCCESS)
		{
			fprintf(stderr, "MPI_Allreduce failed\n");
		}

		/* If total number of student records are larger than ta records,  
		   then fragment student records and replicate ta records. */
		if (stud_num_sum > ta_num_sum)
		{			
			/* Since every process only get a part of ta relation, every process
			   needs to collect all parts from other processes */
			int local_count;

			/* Since each process has different ta number, we need to record them in an array*/
			int ta_num_in_each_process[parts];
			
			status = MPI_Allgather(&tas_count, 1, MPI_INT, ta_num_in_each_process, 1, MPI_INT, MPI_COMM_WORLD);
			
			if(status != MPI_SUCCESS)
			{
				fprintf(stderr, "MPI_Allgather failed\n");
			}

			/* Below two array are sender buffer and receiver buffer. The sender buffer only contain sids of 
			   ta records in this process. The receiver buffer contain sids of ta records from all processes 
			   after calling allgather. */
			int *ta_sid_sender_buff = (int *) malloc(sizeof(int) * tas_count);
			int *ta_sid_receiver_buff = (int *) malloc(sizeof(int) * ta_num_sum);

			for(int i = 0; i<tas_count; i++) 
			{
				ta_sid_sender_buff[i] = tas[i].sid;
			}

			/* calculating displacement */
			int displacement[parts];
			displacement[0] = 0;
			for(int j = 1; j < parts; j++) {
				displacement[j] = displacement[j-1] + ta_num_in_each_process[j-1];
			}

			status = MPI_Allgatherv(ta_sid_sender_buff, tas_count, MPI_INT, ta_sid_receiver_buff, ta_num_in_each_process, displacement, MPI_INT, MPI_COMM_WORLD);
			if(status != MPI_SUCCESS)
			{
				fprintf(stderr, "MPI_Allgatherv failed\n");
			}

			/* After getting all sid from every process, a new ta relation is created, but 
			   note that the records inside only contain sid */
			ta_record *ta_records = (ta_record *) malloc(sizeof(ta_record) * ta_num_sum);
			for(int k = 0; k<ta_num_sum; k++)
			{
				ta_records[k].sid = ta_sid_receiver_buff[k];
			}
			local_count = join_f(students, students_count, ta_records, ta_num_sum);

			//Master process sums up local count from every process
			status = MPI_Allreduce(&local_count, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			if(status != MPI_SUCCESS)
			{
				fprintf(stderr, "MPI_Reduce failed\n");
			}
			free(ta_sid_sender_buff);
			free(ta_sid_receiver_buff);
			free(ta_records);
		}
		else
		{
			/* Total number of ta records is larger than total number of students. However, 
			   ta records is partitioned such that all ta records sid within the partition
			   do not exceed the range of sid of student records. Hence, we do not need to 
			   merge student records. */
			int local_count;
			local_count = join_f(students, students_count, tas, tas_count);
			MPI_Allreduce(&local_count, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		}
	}
	else
	{
		/* Given how ta records are partitioned, we don't need to worry about merging parts
		   from other partitions */
		int local_count;
		local_count = join_f(students, students_count, tas, tas_count);
		MPI_Allreduce(&local_count, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}

	double t_end = MPI_Wtime();

	if (count < 0) goto end;
	if (id == 0) {
		printf("%d\n", count);
		printf("%f\n", (t_end - t_start) * 1000.0);
	}
	result = 0;
end:
	if (students != NULL) free(students);
	if (tas != NULL) free(tas);
	MPI_Finalize();
	return result;
}
