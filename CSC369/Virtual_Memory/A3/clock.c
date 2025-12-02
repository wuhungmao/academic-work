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

int clock_hand;

/* The page to evict is chosen using the CLOCK algorithm.
 * Returns the page frame number (which is also the index in the coremap)
 * for the page that is to be evicted.
 */
int clock_evict()
{
	while (coremap[clock_hand].pte->frame & PAGE_REF)
	{
		// While reference bit is 1, set it to 0
		coremap[clock_hand].pte->frame &= ~PAGE_REF;
		clock_hand = (clock_hand + 1) % memsize;
	}
	// Reference bit is 0
	int ret = clock_hand;
	clock_hand = (clock_hand + 1) % memsize;

	return ret;
}

/* This function is called on every access to a page to update any information
 * needed by the CLOCK algorithm.
 * Input: The page table entry for the page that is being accessed.
 */
void clock_ref(pt_entry_t *pte)
{
	// Set reference bit
	int ref_frame = (int)(pte->frame >> PAGE_SHIFT);
	if (ref_frame < 0 || memsize <= ref_frame)
	{
		perror("Page table entry frame is out of bounds");
		exit(1);
	}

	coremap[ref_frame].pte->frame |= PAGE_REF;
}

/* Initialize any data structures needed for this replacement
 * algorithm.
 */
void clock_init()
{
	clock_hand = 0;
}

/* Cleanup any data structures created in clock_init(). */
void clock_cleanup()
{
	// No need to do anything
}
