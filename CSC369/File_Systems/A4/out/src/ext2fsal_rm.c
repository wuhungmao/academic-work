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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
int32_t ext2_fsal_rm(const char *path)
{
    unsigned int parent_ide_idx;
    char dir_name[EXT2_NAME_LEN + 1];
    struct ext2_inode *inode;
    int ret_val;

    int ide_idx = lookup_inode_by_path(path, &parent_ide_idx, dir_name);

    if (ide_idx < 0)
    {
        return -ide_idx;
    }
    else if (ide_idx > 0)
    {
        // acquire lock for both parent inode and child inode
        lock_resource(0, 0, 0, 0, ide_idx, parent_ide_idx);

        inode = lookup_inode_by_idx(ide_idx);

        // it's a directory
        if ((inode->i_mode & 0xF000) == EXT2_S_IFDIR)
        {
            unlock_resource(0, 0, 0, 0, parent_ide_idx, ide_idx);
            return EISDIR; // found a directory with the same name
        }
        else if (((inode->i_mode & 0xF000) == EXT2_S_IFREG) || ((inode->i_mode & 0xF000) == EXT2_S_IFLNK))
        {
            // rm this file

            // search directory entries stored in parent_ide and find the one matching dir_name(file we want to remove)
            // remove that directory entry and update rec_len of previous directory entry
            ret_val = delete_dir_entry(parent_ide_idx, dir_name);
            CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup);

            // decrement file inode's link count.
            inode->i_links_count--;

            if (inode->i_links_count == 0)
            {
                // if link count is 0, inode is deleted because no hard link points to it
                delete_inode(inode, ide_idx);
            }

            unlock_resource(0, 0, 0, 0, parent_ide_idx, ide_idx);
            return 0;
        }
        else
        {
            unlock_resource(0, 0, 0, 0, parent_ide_idx, ide_idx);
            return EIO;
        }
    }
    else
    {
        return ENOENT;
    }

cleanup:
    unlock_resource(0, 0, 0, 0, parent_ide_idx, ide_idx);
    return ret_val;
}