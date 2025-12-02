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

#ifndef CSC369_E2FS_H
#define CSC369_E2FS_H

#include "ext2.h"
#include <string.h>
#include <stdio.h>
/**
 * TODO: add in here prototypes for any helpers you might need.
 * Implement the helpers in e2fs.c
 */

// .....

// debugging macro. This is for usual helper function that just return error code
#define CHECK_ERROR_CODE(possible_error)                                                             \
    do                                                                                               \
    {                                                                                                \
        int return_val = (possible_error);                                                           \
        if (return_val == ENOENT)                                                                    \
        {                                                                                            \
            return ENOENT;                                                                           \
        }                                                                                            \
        else if (return_val == ENOSPC)                                                               \
        {                                                                                            \
            return ENOSPC;                                                                           \
        }                                                                                            \
        else if (return_val == EISDIR)                                                               \
        {                                                                                            \
            return EISDIR;                                                                           \
        }                                                                                            \
        else if (return_val == EIO)                                                                  \
        {                                                                                            \
            return EIO;                                                                              \
        }                                                                                            \
        else if (return_val == EEXIST)                                                               \
        {                                                                                            \
            return EEXIST;                                                                           \
        }                                                                                            \
        else if (return_val == ENAMETOOLONG)                                                         \
        {                                                                                            \
            return ENAMETOOLONG;                                                                     \
        }                                                                                            \
    } while (0)

// debugging macro. This is for ext2fsal_mkdir, ext2fsal_cp, ext2fsal_rm that needs clean up code
// instead of just returning error
#define CHECK_ERROR_CODE_CLEANUP(error_code, cleanup)                                                      \
    do                                                                                                     \
    {                                                                                                      \
        if ((error_code) == ENOENT)                                                                        \
        {                                                                                                  \
            goto cleanup;                                                                                  \
        }                                                                                                  \
        else if ((error_code) == ENOSPC)                                                                   \
        {                                                                                                  \
            goto cleanup;                                                                                  \
        }                                                                                                  \
        else if ((error_code) == EISDIR)                                                                   \
        {                                                                                                  \
            goto cleanup;                                                                                  \
        }                                                                                                  \
        else if ((error_code) == EIO)                                                                      \
        {                                                                                                  \
            goto cleanup;                                                                                  \
        }                                                                                                  \
        else if ((error_code) == EEXIST)                                                                   \
        {                                                                                                  \
            goto cleanup;                                                                                  \
        }                                                                                                  \
        else if ((error_code) == ENAMETOOLONG)                                                             \
        {                                                                                                  \
            goto cleanup;                                                                                  \
        }                                                                                                  \
    } while (0)

// debugging macro2. For function that return inode idx, data block idx....,
// they cannot return error code directly
// inode index 28 collide with ENOSPC, data block index 2 collides with ENOENT...
#define CHECK_ERROR_INDEX(possible_error)                                                            \
    do                                                                                               \
    {                                                                                                \
        int error = (possible_error);                                                                \
        if (error == 0)                                                                              \
        {                                                                                            \
            return ENOSPC;                                                                           \
        }                                                                                            \
    } while (0)

// debugging macro2. This is for ext2fsal_mkdir, ext2fsal_cp, ext2fsal_rm that needs clean up code
// instead of just returning error
#define CHECK_ERROR_INDEX_CLEANUP(possible_error, cleanup)                                           \
    do                                                                                               \
    {                                                                                                \
        int error = (possible_error);                                                                \
        if (error == 0)                                                                              \
        {                                                                                            \
            goto cleanup;                                                                            \
        }                                                                                            \
    } while (0)

void lock_resource(int lock_sb, int lock_gb, int lock_block_bitmap, int lock_inode_bitmap, unsigned int lock_inode_idx1, unsigned int lock_inode_idx2);
void unlock_resource(int unlock_sb, int unlock_gb, int unlock_block_bitmap, int unlock_inode_bitmap, unsigned int unlock_inode_idx1, unsigned int unlock_inode_idx2);
unsigned int search_in_inode(struct ext2_inode *curr_inode, char *name, int name_len);
struct ext2_inode *lookup_inode_by_idx(int inode_idx);
int lookup_inode_by_path(const char *path, unsigned int *parent_node_idx, char *dir_name);
unsigned int allocate_inode(unsigned short i_mode);
unsigned int allocate_data_block();
int create_dir_entry(unsigned int dir_inode_num, char *name, unsigned int inode_num, unsigned char file_type);
int delete_dir_entry(int dir_inode_num, char *name);
void clear_data_block(struct ext2_inode *target_ide, int target_ide_idx);
int cp_data_block(struct ext2_inode *target_ide, int target_ide_idx, struct ext2_inode *src_inode, int src_ide_idx);
void delete_inode(struct ext2_inode *inode, int ide_idx);
int cp_from_source_file(struct ext2_inode *target_ide, int target_ide_idx, FILE *source_file);
#endif