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

// target size of T1
int p;
extern pt_entry_t *pte_global;

typedef struct arc_node
{
	pt_entry_t *pte;
	struct arc_node *prev;
	struct arc_node *next;
} arc_node;

// Doubly linked list for T1, T2, B1 and B2
typedef struct arc_lst
{
	arc_node *mru;
	arc_node *lru;
	int size;
} arc_lst;

arc_lst *T1;
arc_lst *T2;
arc_lst *B1;
arc_lst *B2;

/* Assert the constraint
 */
void check_constraint()
{
	// Invariants (must hold as list sizes fluctuate) taken from slides:
	// 0 <= |T1| + |T2| + |B1| + |B2| <= 2c, and implicitly 0 <= |T2| + |B2| <= 2c
	// 0 <= |T1| + |B1| <= c
	assert(T1->size + T2->size + B1->size + B2->size <= 2 * memsize);
	assert(T1->size + T2->size + B1->size + B2->size >= 0);
	assert(T2->size + B2->size <= 2 * memsize);
	assert(T2->size + B2->size >= 0);
	assert(T1->size + B1->size <= memsize);
	assert(T1->size + B1->size >= 0);
}

// Returns the arc_node of pte if pte is in arc_lst, otherwise return NULL
arc_node *return_node(pt_entry_t *pte, arc_lst *arc_lst)
{
	arc_node *curr = arc_lst->mru;

	for (int i = 0; i < arc_lst->size; ++i)
	{
		if (curr->pte == pte)
		{
			return curr;
		}
		curr = curr->next;
	}

	return NULL;
}

/* Extracts the given node from the list, and decrements the list size
 * Precondition: Node is in the list
 */
void extract_node(arc_node *node, arc_lst *list)
{
	// Remove node references from list
	if (node->prev)					   // Not first node (Not MRU node)
		node->prev->next = node->next; // Link previous to next
	else							   // Is MRU node
		list->mru = node->next;		   // Link new MRU to next

	if (node->next)					   // Not last node (Not LRU node)
		node->next->prev = node->prev; // Link next to previous
	else							   // Is LRU node
		list->lru = node->prev;		   // Link new LRU to prev

	list->size--;

	// Remove list references from node
	node->prev = NULL;
	node->next = NULL;
}

/* Insert the given node into the list as the MRU node, increments the list size
 * If the list is empty, also set the LRU reference as the inserted node
 */
void insert_node(arc_node *node, arc_lst *list)
{
	// Add list reference to node
	node->prev = NULL;
	node->next = list->mru;

	// Add node reference to list
	if (list->mru != NULL)
	{
		// If list is not empty
		list->mru->prev = node;
	}
	else
	{
		// List is empty
		list->lru = node;
	}
	list->mru = node;
	list->size++;
}

/* Called by case 2 and 3
 */
void replace(arc_node **evicted_node)
{
	// Implement replace function from slides
	// Extract the evicted node from T1 or T2
	if (T1->size > p)
	{
		// T1 is too large
		*evicted_node = T1->lru;
		extract_node(*evicted_node, T1);

		if (B1->size != 0)
		{
			arc_node *ghost_node = B1->lru; // Repurpose extracted B1 node to be evicted T1 node's new B1 node
			extract_node(ghost_node, B1);
			insert_node(ghost_node, B1);
		}
	}
	else
	{
		// T2 is too large
		*evicted_node = T2->lru;
		extract_node(*evicted_node, T2);

		if (B2->size != 0)
		{
			arc_node *ghost_node = B2->lru; // Repurpose extracted B2 node to be evicted T2 node's new B2 node
			extract_node(ghost_node, B2);
			insert_node(ghost_node, B2);
		}
	}
	check_constraint();
}

// return 0 if not found, 1 if in T1, 2 if in T2, 3 if in B1, 4 if in B2
int find_node(pt_entry_t *pte)
{
	if (return_node(pte, T1) != NULL)
	{
		return 1;
	}
	else if (return_node(pte, T2) != NULL)
	{
		return 2;
	}
	else if (return_node(pte, B1) != NULL)
	{
		return 3;
	}
	else if (return_node(pte, B2) != NULL)
	{
		return 4;
	}
	else
	{
		return 0;
	}
}

// return min
int min(int a, int b)
{
	int min = ((a > b) ? b : a);
	return min;
}

// return max
int max(int a, int b)
{
	int max = ((a > b) ? a : b);
	return max;
}

/* The page to evict is chosen using the ARC algorithm.
 * Returns the page frame number (which is also the index in the coremap)
 * for the page that is to be evicted.
 */
int arc_evict()
{
	int delta;
	// searching to see if this new pte was stored in B1 or B2 or complete miss
	if (find_node(pte_global) == 3)
	{
		// case 2 from slides
		// pte stored in B1
		delta = (B1->size >= B2->size) ? 1 : (B2->size / B1->size);
		p = min(p + delta, memsize);
		// call replace
		arc_node *evicted_node;
		replace(&evicted_node);
		int evicted_frame = (int)(evicted_node->pte->frame >> PAGE_SHIFT);
		free(evicted_node);
		return evicted_frame;
	}
	else if (find_node(pte_global) == 4)
	{
		// case 3 from slides
		// pte stored in B2
		delta = (B2->size >= B1->size) ? 1 : (B1->size / B2->size);
		p = max(p - delta, 0);
		// call replace
		arc_node *evicted_node;
		replace(&evicted_node);
		int evicted_frame = (int)(evicted_node->pte->frame >> PAGE_SHIFT);
		free(evicted_node);
		return evicted_frame;
	}
	else
	{
		// case 4
		// access is a complete miss
		// no adjustment for p
		assert(T1->size + B1->size <= memsize);
		assert(T1->size + B1->size >= 0);
		if (T1->size + B1->size == memsize)
		{
			// case A, |T1| + |B1| = c
			if (T1->size < memsize)
			{
				// free B1 LRU
				arc_node *b1_lru = B1->lru;
				B1->lru = b1_lru->prev;
				b1_lru->prev = NULL;
				b1_lru->next = NULL;
				if (B1->lru != NULL)
				{
					B1->lru->next = NULL;
				}
				free(b1_lru);
				B1->size--;

				// apply replace
				arc_node *evicted_node;
				replace(&evicted_node);

				int evicted_frame = (int)(evicted_node->pte->frame >> PAGE_SHIFT);
				free(evicted_node);
				check_constraint();
				return evicted_frame;
			}
			else
			{
				// delete page in the LRU(T1) slot (and evict the page)
				arc_node *t1_lru = T1->lru;
				T1->lru = t1_lru->prev;
				t1_lru->prev = NULL;
				t1_lru->next = NULL;
				if (T1->lru != NULL)
				{
					T1->lru->next = NULL;
				}
				int evicted_frame = (int)(t1_lru->pte->frame >> PAGE_SHIFT);
				free(t1_lru);
				T1->size--;
				check_constraint();
				return evicted_frame;
			}
		}
		else
		{
			// case B, |T1| + |B1| < c
			if (T1->size + T2->size + B1->size + B2->size >= memsize)
			{
				// if |T1| + |T2| + |B1| + |B2| = 2c, delete LRU(B2), then Apply REPLACE
				if (T1->size + T2->size + B1->size + B2->size == 2 * memsize)
				{
					// in B2
					// delete B2 lru node
					if (B2->lru->prev != NULL)
					{
						arc_node *lru = B2->lru;
						B2->lru = B2->lru->prev;
						free(lru);
						B2->lru->next = NULL;
						B2->size--;
					}
					else
					{
						B2->lru = NULL;
						if (B2->size > 0)
						{
							B2->size--;
						}
					}
				}
				check_constraint();

				// apply replace
				arc_node *evicted_node;
				replace(&evicted_node);

				int evicted_frame = (int)(evicted_node->pte->frame >> PAGE_SHIFT);
				free(evicted_node);

				return evicted_frame;
			}
			// if |T1| + |T2| + |B1| + |B2| < c, do nothing
			// Never call evict if |T1| + |T2| < c, so we never encounter this
			return -1;
		}
	}
}

void move_to_T2(pt_entry_t *pte, int ret)
{
	// find arc_node for this pte
	if (ret == 1)
	{
		// case 1 (1)
		// in t1
		arc_node *target = return_node(pte, T1);
		extract_node(target, T1);
		insert_node(target, T2);
	}
	else if (ret == 2)
	{
		// case 1 (2)
		// in t2
		arc_node *target = return_node(pte, T2);
		extract_node(target, T2);
		insert_node(target, T2);
	}
	else if (ret == 3)
	{
		// case 2.c
		// in B1
		arc_node *target = return_node(pte, B1);
		extract_node(target, B1);
		insert_node(target, T2);
	}
	else if (ret == 4)
	{
		// case 3.c
		// in B2
		arc_node *target = return_node(pte, B2);
		extract_node(target, B2);
		insert_node(target, T2);
	}
}

/* This function is called on each access to a page to update any information
 * needed by the ARC algorithm.
 * Input: The page table entry for the page that is being accessed.
 */
void arc_ref(pt_entry_t *pte)
{
	int ret = find_node(pte);
	if (ret)
	{
		// case 1, 2, 3
		move_to_T2(pte, ret);
	}
	else
	{
		// case 4
		arc_node *new_node = (arc_node *)malloc(sizeof(arc_node));
		if (!new_node)
		{
			perror("Failed to allocate node");
			exit(1);
		}
		new_node->pte = pte;
		insert_node(new_node, T1);
	}
	check_constraint();
}

/* Initializes any data structures needed for this
 * replacement algorithm.
 */
void arc_init()
{
	// initialize T1, T2, B1, B2
	T1 = (arc_lst *)malloc(sizeof(arc_lst));
	T1->size = 0;
	T2 = (arc_lst *)malloc(sizeof(arc_lst));
	T2->size = 0;
	B1 = (arc_lst *)malloc(sizeof(arc_lst));
	B1->size = 0;
	B2 = (arc_lst *)malloc(sizeof(arc_lst));
	B2->size = 0;

	if (T1 == NULL || T2 == NULL || B1 == NULL || B2 == NULL)
	{
		perror("Failed to allocate memory for ARC");
		exit(1);
	}

	p = 0;
}

/* Free a list
 */
void free_list(arc_lst *lst)
{
	arc_node *curr = lst->mru;
	arc_node *to_free = curr;
	while (curr != NULL)
	{
		curr = curr->next;
		free(to_free);
		to_free = curr;
	}
	free(lst);
}

/* Cleanup any data structures created in arc_init(). */
void arc_cleanup()
{
	free_list(T1);
	free_list(T2);
	free_list(B1);
	free_list(B2);
}
