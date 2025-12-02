/*
 * This code is provided solely for the personal and private use of students
 * taking the CSC369H course at the University of Toronto. Copying for purposes
 * other than this use is expressly prohibited. All forms of distribution of
 * this code, including but not limited to public repositories on GitHub,
 * GitLab, Bitbucket, or any other online platform, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Andrew Peterson, Karen Reid, Alexey Khrabrov, Vladislav Sytchenko
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2025 Bogdan Simion, Karen Reid
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "pagetable.h"

extern int memsize;
extern bool debug;
extern struct frame *coremap;

int earliest_frame; // Index of the earliest frame in coremap

/* Page to evict is chosen using the FIFO algorithm.
 * Returns the page frame number (which is also the index in the coremap)
 * for the page that is to be evicted.
 */
int fifo_evict()
{
	// In pagetable.c allocate_frame(), frames are allocated sequentially starting from 0
	// This means the first memsize number of frames are already sorted by first-in in coremap
	// We only need to circularly iterate through the indexes
	int evicted_frame = earliest_frame;
	earliest_frame = (earliest_frame + 1) % memsize;
	return evicted_frame;
}

/* This function is called on each access to a page to update any information
 * needed by the FIFO algorithm.
 * Input: The page table entry for the page that is being accessed.
 */
void fifo_ref(pt_entry_t *pte)
{
	// No need to do anything
	(void)pte;
}

/* Initialize any data structures needed for this
 * replacement algorithm
 */
void fifo_init()
{
	earliest_frame = 0;
}

/* Cleanup any data structures created in fifo_init(). */
void fifo_cleanup()
{
	// No need to do anything
}
