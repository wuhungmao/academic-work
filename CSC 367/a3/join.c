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
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "join.h"

int join_nested(const student_record *students, int students_count, const ta_record *tas, int tas_count)
{
	assert(students != NULL);
	assert(tas != NULL);
	int count = 0;
	for(int student_ind = 0; student_ind < students_count; student_ind++) 
	{
		for(int ta_ind = 0; ta_ind < tas_count; ta_ind++) 
		{
			if ((students[student_ind].gpa > 3.0) && (tas[ta_ind].sid == students[student_ind].sid))
			{
				count++;
			}
		}
	}
	return count;
}

// Assumes that the records in both tables are already sorted by sid
int join_merge(const student_record *students, int students_count, const ta_record *tas, int tas_count)
{
	assert(students != NULL);
	assert(tas != NULL);

	int stud_ind = 0;
	int ta_ind = 0;
	int count = 0;
	while(stud_ind < students_count && ta_ind < tas_count) 
	{
	    if (students[stud_ind].sid > tas[ta_ind].sid)
		{
			ta_ind++;
		}
		else if (students[stud_ind].sid < tas[ta_ind].sid)
		{
			stud_ind++;
		}
	    else
		{
	        /* found a match for equi-join */
			if (students[stud_ind].gpa>3.0) 
			{
				count++;
			}
			int ta_ind_dup = ta_ind+1;
	        while(ta_ind_dup < tas_count && students[stud_ind].sid == tas[ta_ind_dup].sid && students[stud_ind].gpa>3.0) 
			{
				count++;
				ta_ind_dup++;
			}
			ta_ind=ta_ind_dup;
			stud_ind++;
		}
	}
	return count;
}

int hash(int key, int students_count) 
{
	return key%students_count;
}

typedef struct _hash_table_t {
	int *size_of_buck;
	int *capa_of_buck;
	student_record **buckets;
}hash_table;

int join_hash(const student_record *students, int students_count, const ta_record *tas, int tas_count)
{
	assert(students != NULL);
	assert(tas != NULL);
	
	hash_table table;
	table.buckets = (student_record **) malloc(sizeof(student_record *) * students_count);
	table.capa_of_buck = (int *) malloc(sizeof(int) * students_count);
	table.size_of_buck = (int *) malloc(sizeof(int) * students_count);
	for(int buck_ind = 0; buck_ind < students_count; buck_ind++)
	{
		table.buckets[buck_ind] = (student_record *) malloc(sizeof(student_record) * 10);

		if (table.buckets[buck_ind] == NULL) {
			fprintf(stderr, "malloc error for bucket %d\n", buck_ind);
			break; // Exit the loop
		}
		table.capa_of_buck[buck_ind] = 10;
		table.size_of_buck[buck_ind] = 0;
	}

	int hash_val_stu;
	int curr_empty_slot;
	for(int stud_ind = 0; stud_ind < students_count; stud_ind++)
	{
		hash_val_stu = hash(students[stud_ind].sid, students_count);
		curr_empty_slot = table.size_of_buck[hash_val_stu];
		table.buckets[hash_val_stu][curr_empty_slot] = students[stud_ind];
		table.size_of_buck[hash_val_stu]+=1;
 
		//realloc if buckets does not contain enough memory
		if(table.size_of_buck[hash_val_stu] >= table.capa_of_buck[hash_val_stu])
		{
			int new_capacity = table.capa_of_buck[hash_val_stu] * 2;
			table.buckets[hash_val_stu] = realloc(table.buckets[hash_val_stu], new_capacity * sizeof(student_record));
			table.capa_of_buck[hash_val_stu] = new_capacity;
		}
	}

	/* Lookup in S */
	int hash_val_ta;
	int count = 0;
	for(int ta_ind = 0; ta_ind < tas_count; ta_ind++) 
	{
		hash_val_ta = hash(tas[ta_ind].sid, students_count);
		if(table.size_of_buck[hash_val_ta] != 0) 
		{
			int bucket_size = table.size_of_buck[hash_val_ta];
			for(int pos=0;pos<bucket_size;pos++) 
			{
				if(table.buckets[hash_val_ta][pos].sid == tas[ta_ind].sid && table.buckets[hash_val_ta][pos].gpa > 3.0) 
				{
					count++;
					break;
				}
			}
		}
	}
	
	for(int i=0; i < students_count; i++) {
		free(table.buckets[i]);
	}
	free(table.buckets);
	free(table.capa_of_buck);
	free(table.size_of_buck);
	return count;
}
