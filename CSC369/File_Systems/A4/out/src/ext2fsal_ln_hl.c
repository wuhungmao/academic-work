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

int32_t ext2_fsal_ln_hl(const char *src,
                        const char *dst)
{
    unsigned int src_dir_i; // Index of src parent directory inode
    char src_name[EXT2_NAME_LEN];
    int ret_val;

    // src inode number, 0 if the src does not exists, or -ENOENT if path is invalid
    int src_inode_i = lookup_inode_by_path(src, &src_dir_i, src_name);

    if (src_inode_i == 0 || src_inode_i == -ENOENT)
    {
        return ENOENT; // The source name does not exist or contains an invalid path
    }
    if (src_inode_i < 0)
    {
        return -src_inode_i;
    }

    // Locate final directory inode
    unsigned int dst_dir_i = 0;
    char link_name[EXT2_NAME_LEN];
    int dst_link_i = lookup_inode_by_path(dst, &dst_dir_i, link_name);

    lock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);
    struct ext2_inode *src_inode = lookup_inode_by_idx(src_inode_i);
    // Check src inode status
    if ((src_inode->i_mode & 0xF000) == EXT2_S_IFDIR)
    { // If src refers to a directory
        unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);
        return EISDIR;
    }

    // Check dst path status
    if (dst_link_i < 0)
    { // If dst is an invalid path
        unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);
        return ENOENT;
    } // Else dst parent directory exists

    if (src_inode_i == dst_dir_i)
    { // This case should never be possible, unless lookup fails
        unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);
        return ENOENT;
    }

    struct ext2_inode *dst_dir_inode = lookup_inode_by_idx(dst_dir_i);
    // Check dst inode status
    if ((dst_dir_inode->i_mode & 0xF000) == EXT2_S_IFREG)
    { // The dst directory refers to a file
        unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);

        return ENOENT;
    }

    if (dst_link_i > 0)
    { // If dst name already exists
        unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);
        struct ext2_inode *dst_inode = lookup_inode_by_idx(dst_link_i);
        if ((dst_inode->i_mode & 0xF000) == EXT2_S_IFDIR)
        {

            return EISDIR;
        }
        // Else the inode is a file (including hard-link, or soft-link)

        return EEXIST;
    }

    // All arguments are valid for operations
    ret_val = create_dir_entry(dst_dir_i, link_name, src_inode_i, EXT2_FT_REG_FILE);
    CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup);
    src_inode->i_links_count++;
    unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);

    return 0;

cleanup:
    unlock_resource(0, 0, 0, 0, src_inode_i, dst_dir_i);
    return ret_val;
}