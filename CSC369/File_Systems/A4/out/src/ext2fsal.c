/*
 *------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC369H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2025 MCS @ UTM
 * -------------
 */

// copy from e2fs.c, not sure if this is allowed
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "ext2fsal.h"
#include "e2fs.h"

unsigned char *disk;
struct ext2_super_block *sb;
struct ext2_inode *inode_table;
struct ext2_group_desc *block_group_desc;
unsigned char *block_bitmap;
unsigned char *inode_bitmap;

// Global Locks
pthread_mutex_t sb_mutex;
pthread_mutex_t group_desc_mutex;
pthread_mutex_t block_bitmap_mutex;
pthread_mutex_t inode_bitmap_mutex;
pthread_mutex_t inode_mutex[INODE_COUNT];

void init_all_mutexes(void)
{
    if (pthread_mutex_init(&sb_mutex, NULL) != 0)
    {
        perror("Failed to initialize sb_mutex");
        exit(1);
    }
    if (pthread_mutex_init(&group_desc_mutex, NULL) != 0)
    {
        perror("Failed to initialize group_desc_mutex");
        exit(1);
    }
    if (pthread_mutex_init(&block_bitmap_mutex, NULL) != 0)
    {
        perror("Failed to initialize block_bitmap_mutex");
        exit(1);
    }
    if (pthread_mutex_init(&inode_bitmap_mutex, NULL) != 0)
    {
        perror("Failed to initialize inode_bitmap_mutex");
        exit(1);
    }

    for (int i = 0; i < INODE_COUNT; ++i)
    {
        if (pthread_mutex_init(&inode_mutex[i], NULL) != 0)
        {
            perror("Failed to initialize inode_mutex");
            exit(1);
        }
    }
}

void destroy_all_mutexes(void)
{
    pthread_mutex_destroy(&sb_mutex);

    pthread_mutex_destroy(&group_desc_mutex);
    pthread_mutex_destroy(&block_bitmap_mutex);
    pthread_mutex_destroy(&inode_bitmap_mutex);
    for (int i = 0; i < INODE_COUNT; ++i)
    {
        pthread_mutex_destroy(&inode_mutex[i]);
    }
}

void ext2_fsal_init(const char *image)
{
    int fd = open(image, O_RDWR);

    disk = mmap(NULL, 128 * EXT2_BLOCK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (disk == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }

    // super block
    sb = (struct ext2_super_block *)(disk + EXT2_BLOCK_SIZE);

    // group descriptor
    block_group_desc = (struct ext2_group_desc *)(disk + 2 * EXT2_BLOCK_SIZE);

    // block bitmap
    block_bitmap = (disk + EXT2_BLOCK_SIZE * block_group_desc->bg_block_bitmap);

    // inode bitmap
    inode_bitmap = (disk + EXT2_BLOCK_SIZE * block_group_desc->bg_inode_bitmap);

    // inode table
    inode_table = (struct ext2_inode *)(disk + EXT2_BLOCK_SIZE * block_group_desc->bg_inode_table);

    init_all_mutexes();
}

void ext2_fsal_destroy()
{
    destroy_all_mutexes();
}