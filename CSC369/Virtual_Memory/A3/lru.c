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

// Double linked list implementation
typedef struct lru_node
{
	int frame;
	struct lru_node *prev; // Less recently used
	struct lru_node *next; // Most recently used
} lru_node;

lru_node *least_recent_node; // Pointer to the first element of a list of frames sorted by lru
lru_node *most_recent_node;	 // Pointer to the last element of a list of frames sorted by lru

lru_node *array;

/* The page to evict is chosen using the accurate LRU algorithm.
 * Returns the page frame number (which is also the index in the coremap)
 * for the page that is to be evicted.
 */
int lru_evict()
{
	if (least_recent_node == NULL)
	{
		perror("No frames recorded in LRU list");
		exit(1);
	}

	// Extract the first node from list
	lru_node *evicted_node = least_recent_node;
	least_recent_node = least_recent_node->next;
	if (least_recent_node != NULL)
	{
		// In case memsize == 1
		least_recent_node->prev = NULL;
	}

	// cut the ties for the node to be evicted
	evicted_node->prev = NULL;
	evicted_node->next = NULL;
	int evicted_frame = evicted_node->frame;

	return evicted_frame;
}

/* This function is called on each access to a page to update any information
 * needed by the LRU algorithm.
 * Input: The page table entry for the page that is being accessed.
 */
void lru_ref(pt_entry_t *pte)
{
	int ref_frame = (int)(pte->frame >> PAGE_SHIFT);
	if (ref_frame < 0 || memsize <= ref_frame)
	{
		perror("Page table entry frame is out of bounds");
		exit(1);
	}
	lru_node *ref_node = &(array[ref_frame]);

	// Extract the node references from list (if in list)
	if (ref_node->prev != NULL)
	{
		ref_node->prev->next = ref_node->next;
	}

	if (ref_node->next != NULL)
	{
		ref_node->next->prev = ref_node->prev;
	}

	// If the removed node is the list head or tail
	if (ref_node == least_recent_node)
	{
		least_recent_node = ref_node->next; // Make second least recent node the new head
	}

	if (ref_node == most_recent_node)
	{
		most_recent_node = ref_node->prev; // Make second most recent node the new tail
	}

	// Insert to end of list
	ref_node->prev = most_recent_node;
	ref_node->next = NULL;
	most_recent_node = ref_node;

	// If this is the only node in the list
	if (least_recent_node == NULL)
	{
		least_recent_node = ref_node;
	}

	if (ref_node->prev != NULL)
	{
		ref_node->prev->next = ref_node;
	}
}

/* Initialize any data structures needed for this
 * replacement algorithm
 */
void lru_init()
{
	// Initialize LRU and MRU node
	least_recent_node = NULL;
	most_recent_node = NULL;

	array = malloc(memsize * sizeof(lru_node));
	if (array == NULL)
	{
		perror("Failed to allocate memory for LRU");
		exit(1);
	}

	// initialize every node
	for (int i = 0; i < memsize; ++i)
	{
		array[i].frame = i;
	}
}

/* Cleanup any data structures created in lru_init(). */
void lru_cleanup()
{
	free(array);
	least_recent_node = NULL;
	most_recent_node = NULL;
}
