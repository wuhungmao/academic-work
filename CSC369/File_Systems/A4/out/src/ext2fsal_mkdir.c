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
#include <libgen.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

extern unsigned char *disk;
extern struct ext2_super_block *sb;
extern struct ext2_inode *inode_table;
extern struct ext2_group_desc *block_group_desc;
extern unsigned char *block_bitmap;
extern unsigned char *inode_bitmap;

int32_t ext2_fsal_mkdir(const char *path)
{
    unsigned int parent_ide_idx = 0;
    char new_dir_name[EXT2_NAME_LEN + 1];
    unsigned int new_inode_idx;
    int ret_val;
    struct ext2_inode *parent_inode;

    int ide_idx = lookup_inode_by_path(path, &parent_ide_idx, new_dir_name);

    // Check path status
    if (ide_idx < 0)
    {
        // error occurred during traversal
        return -ide_idx;
    }
    else if (ide_idx > 0)
    { // If name is already used
        // Check inode status
        lock_resource(0, 0, 0, 0, ide_idx, 0);

        struct ext2_inode *inode = lookup_inode_by_idx(ide_idx);

        // directory already exists
        if ((inode->i_mode & 0xF000) == EXT2_S_IFDIR)
        {
            unlock_resource(0, 0, 0, 0, ide_idx, 0);
            return EEXIST;
        }
        // provided directory is a regular file
        else if ((inode->i_mode & 0xF000) == EXT2_S_IFREG)
        {
            unlock_resource(0, 0, 0, 0, ide_idx, 0);
            return ENOENT;
        }
        // provided directory is a symbolic link
        else if ((inode->i_mode & 0xF000) == EXT2_S_IFLNK)
        {
            unlock_resource(0, 0, 0, 0, ide_idx, 0);
            return EEXIST;
        }
        else
        {
            unlock_resource(0, 0, 0, 0, ide_idx, 0);
            return EIO;
        }
    }
    else
    {
        // directory does not exist, create it
        // allocate new inode
        // i_mode: directory with 755 permissions
        new_inode_idx = allocate_inode(EXT2_S_IFDIR | 0x1ED);

        CHECK_ERROR_INDEX_CLEANUP(new_inode_idx, cleanup_parent_lock);

        lock_resource(0, 0, 0, 0, new_inode_idx, parent_ide_idx);

        // Write . and .. into new directory data block
        char *dot = ".";
        char *dot_dot = "..";
        ret_val = create_dir_entry(new_inode_idx, dot, new_inode_idx, EXT2_FT_DIR);
        CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_both);

        ret_val = create_dir_entry(new_inode_idx, dot_dot, parent_ide_idx, EXT2_FT_DIR);
        CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_both);

        // update parent directory to add new entry
        ret_val = create_dir_entry(parent_ide_idx, new_dir_name, new_inode_idx, EXT2_FT_DIR);
        CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_both);

        // update link count
        parent_inode = lookup_inode_by_idx(parent_ide_idx);
        parent_inode->i_links_count++;

        unlock_resource(0, 0, 0, 0, parent_ide_idx, new_inode_idx);

        return 0; // success
    }

cleanup_parent_lock:
    return ENOSPC;

cleanup_both:
    // deallocate allocated inode
    struct ext2_inode *inode_to_be_destroy = lookup_inode_by_idx(new_inode_idx);
    delete_inode(inode_to_be_destroy, new_inode_idx);
    unlock_resource(0, 0, 0, 0, parent_ide_idx, new_inode_idx);
    return ret_val;
}