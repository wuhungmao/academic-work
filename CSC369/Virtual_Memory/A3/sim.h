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

#ifndef __SIM_H__
#define __SIM_H__

#include "pagetable.h"


/* Simulated physical memory page frame size */
#define SIMPAGESIZE 16

extern int memsize;
extern bool debug;

extern size_t hit_count;
extern size_t miss_count;
extern size_t ref_count;
extern size_t evict_clean_count;
extern size_t evict_dirty_count;

/* We simulate physical memory with a large array of bytes */
extern char *physmem;

extern void (*ref_func)(pt_entry_t *);
extern int (*evict_func)(void);


#endif /* __SIM_H__ */
