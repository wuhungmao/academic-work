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

#include "ext2fsal.h"
#include "e2fs.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int32_t ext2_fsal_ln_sl(const char *src,
                        const char *dst)
{
    int ret_val;
    unsigned short PATH_MAX = 4096;

    int src_path_len = strlen(src);
    // Check path status
    if (src_path_len > PATH_MAX)
    {
        return ENAMETOOLONG;
    }

    // Locate dst parent directory inode
    unsigned int dst_dir_i;
    char link_name[EXT2_NAME_LEN];
    int dst_link_i = lookup_inode_by_path(dst, &dst_dir_i, link_name);

    if (dst_link_i < 0)
    {
        return -dst_link_i;
    }
    lock_resource(0, 0, 0, 0, dst_dir_i, 0);
    struct ext2_inode *dst_dir_inode = lookup_inode_by_idx(dst_dir_i);
    if ((dst_dir_inode->i_mode & 0xF000) == EXT2_S_IFREG)
    { // If dst directory refers to a file
        unlock_resource(0, 0, 0, 0, dst_dir_i, 0);
        return ENOENT;
    }

    if (dst_link_i > 0)
    { // If file with that name already exists
        struct ext2_inode *dst_inode = lookup_inode_by_idx(dst_link_i);
        if ((dst_inode->i_mode & 0xF000) == EXT2_S_IFDIR)
        {
            unlock_resource(0, 0, 0, 0, dst_dir_i, 0);
            return EISDIR;
        }
        unlock_resource(0, 0, 0, 0, dst_dir_i, 0);
        return EEXIST;
    }
    unlock_resource(0, 0, 0, 0, dst_dir_i, 0);
    // Allocate a new inode
    // All arguments are valid for operations
    dst_link_i = allocate_inode(EXT2_S_IFLNK);

    CHECK_ERROR_INDEX_CLEANUP(dst_link_i, cleanup_allocation1);

    lock_resource(0, 0, 0, 0, dst_link_i, dst_dir_i);
    struct ext2_inode *dst_link_inode = lookup_inode_by_idx(dst_link_i);
    dst_link_inode->i_size = src_path_len;

    // Allocate and write to data blocks
    const char *write_start_ptr = src;
    int remaining = src_path_len;
    for (int i = 0; i < 4 && remaining > 0; ++i)
    { // Allocate at most four blocks = ceil(PATH_MAX / EXT2_BLOCK_SIZE)
        // Allocate a block
        unsigned int data_block_i = allocate_data_block();

        CHECK_ERROR_INDEX_CLEANUP(data_block_i, cleanup_allocation2);
        dst_link_inode->i_block[i] = data_block_i;
        dst_link_inode->i_blocks += 2; // 2 disk sectors per data block
        // Write the path to the block
        unsigned char *data_block = (unsigned char *)(disk + EXT2_BLOCK_SIZE * data_block_i);

        // The minimum between the length of the unwritten path and data block size
        int char_to_write = remaining < EXT2_BLOCK_SIZE ? remaining : EXT2_BLOCK_SIZE;

        memcpy(data_block, write_start_ptr, char_to_write); // Write path up to the end of the block
        write_start_ptr += char_to_write;
        remaining -= char_to_write; // Calculate length of remaining unwritten path
    }

    ret_val = create_dir_entry(dst_dir_i, link_name, dst_link_i, EXT2_FT_SYMLINK);
    CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_delete_inode);
    dst_link_inode->i_links_count++;
    unlock_resource(0, 0, 0, 0, dst_dir_i, dst_link_i);
    return 0;

cleanup_allocation1:
    return ENOSPC;

cleanup_allocation2:
    if (dst_dir_i != 0)
        unlock_resource(0, 0, 0, 0, dst_dir_i, 0);
    if (dst_link_i != 0)
        unlock_resource(0, 0, 0, 0, dst_link_i, 0);
    return ENOSPC;

cleanup_delete_inode:
    delete_inode(dst_link_inode, dst_link_i);
    unlock_resource(0, 0, 0, 0, dst_dir_i, dst_link_i);
    return ret_val;
}
