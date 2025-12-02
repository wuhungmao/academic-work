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

int32_t ext2_fsal_cp(const char *src,
                     const char *dst)
{

    // look for source in native operating system
    FILE *source_file = fopen(src, "rb");

    // source file does not exist on native os, guaranteed to not be a directory by assignment handout
    if (source_file == NULL)
    {
        return ENOENT;
    }

    char *source_file_name;
    char *last_slash_source_name = strrchr(src, '/');

    if (last_slash_source_name != NULL)
    {
        source_file_name = last_slash_source_name + 1;
    }
    else
    {
        // No slash found, the whole path is the source name
        source_file_name = strdup(src);
    }

    unsigned int dst_parent_ide_idx;
    char dest_name[EXT2_NAME_LEN + 1];
    // find the destination inode
    int dst_ide_idx = lookup_inode_by_path(dst, &dst_parent_ide_idx, dest_name);

    struct ext2_inode *dst_inode;
    struct ext2_inode *target_ide;
    int target_ide_idx = 0;
    int ret_val;

    // Check dst path status
    if (dst_ide_idx < 0)
    { // If dst is an invalid path
        fclose(source_file);
        return ENOENT;
    }

    if (dst_ide_idx == 0)
    { // If dst target does not exist
        // allocate new inode and copy over source file
        target_ide_idx = allocate_inode(EXT2_S_IFREG | 0x1ED);
        CHECK_ERROR_INDEX_CLEANUP(target_ide_idx, cleanup_allocate2);

        lock_resource(0, 0, 0, 0, target_ide_idx, dst_parent_ide_idx);
        target_ide = lookup_inode_by_idx(target_ide_idx);
        // update parent inode directory entry
        ret_val = create_dir_entry(dst_parent_ide_idx, dest_name, target_ide_idx, EXT2_FT_REG_FILE);
        CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_delete_inode1);
        ret_val = cp_from_source_file(target_ide, target_ide_idx, source_file);
        CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_delete_inode1);
        unlock_resource(0, 0, 0, 0, target_ide_idx, dst_parent_ide_idx);
        fclose(source_file);
        return 0;
    }

    // destination inode is valid
    lock_resource(0, 0, 0, 0, dst_ide_idx, 0);
    dst_inode = lookup_inode_by_idx(dst_ide_idx);

    // Check inode status
    // destination is a symbolic link, so we return eexist
    if ((dst_inode->i_mode & 0xF000) == EXT2_S_IFLNK)
    {
        unlock_resource(0, 0, 0, 0, dst_ide_idx, 0);
        fclose(source_file);
        return EEXIST;
    }

    // destination inode is a file
    if ((dst_inode->i_mode & 0xF000) == EXT2_S_IFREG)
    {
        // A very special and tricky case, cp /afile /bfile/ where afile and bfile are both files,
        // it should return ENOENT because of the last slash. but lookup_path_by_idx strip trailing slash
        int len = strlen(dst);
        if (len > 0 && dst[len - 1] == '/')
        {
            unlock_resource(0, 0, 0, 0, dst_ide_idx, 0);
            fclose(source_file);
            return ENOENT;
        }

        // need to clear the data block owned by destination inode
        clear_data_block(dst_inode, dst_ide_idx);
        // then copy over the source file
        ret_val = cp_from_source_file(dst_inode, dst_ide_idx, source_file);
        CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup);
        unlock_resource(0, 0, 0, 0, dst_ide_idx, 0);
    }

    // destination inode is a directory
    if ((dst_inode->i_mode & 0xF000) == EXT2_S_IFDIR)
    {
        target_ide_idx = search_in_inode(dst_inode, source_file_name, strlen(source_file_name));

        // Found inode with same name as source file within destination directory
        if (target_ide_idx != 0)
        {
            lock_resource(0, 0, 0, 0, target_ide_idx, 0);
            target_ide = lookup_inode_by_idx(target_ide_idx);

            if ((target_ide->i_mode & 0xF000) == EXT2_S_IFREG)
            {
                // Free Old Blocks and update block bitmap
                clear_data_block(target_ide, target_ide_idx);
                // Use the existing inode to hold same data as source file
                ret_val = cp_from_source_file(target_ide, target_ide_idx, source_file);
                CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup);
                unlock_resource(0, 0, 0, 0, target_ide_idx, dst_ide_idx);
            }
            else if ((target_ide->i_mode & 0xF000) == EXT2_S_IFDIR)
            {
                // directory with same name already exist
                unlock_resource(0, 0, 0, 0, target_ide_idx, dst_ide_idx);
                return EISDIR;
            }
            else if ((target_ide->i_mode & 0xF000) == EXT2_S_IFLNK)
            {
                // symbolic link with same name already exist
                unlock_resource(0, 0, 0, 0, target_ide_idx, dst_ide_idx);
                return EEXIST;
            }
            else
            {
                // unknown inode type
                unlock_resource(0, 0, 0, 0, target_ide_idx, dst_ide_idx);
                return EIO;
            }
        }
        else
        {
            // cannot find inode with same name
            target_ide_idx = allocate_inode(EXT2_S_IFREG | 0x1ED);

            CHECK_ERROR_INDEX_CLEANUP(target_ide_idx, cleanup_allocate1);
            lock_resource(0, 0, 0, 0, target_ide_idx, 0);
            target_ide = lookup_inode_by_idx(target_ide_idx);

            // update parent inode directory entry, copy data from source file
            ret_val = create_dir_entry(dst_ide_idx, source_file_name, target_ide_idx, EXT2_FT_REG_FILE);
            CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_delete_inode2);
            ret_val = cp_from_source_file(target_ide, target_ide_idx, source_file);
            CHECK_ERROR_CODE_CLEANUP(ret_val, cleanup_delete_inode2);
            unlock_resource(0, 0, 0, 0, target_ide_idx, dst_ide_idx);
        }
    }

    fclose(source_file);
    return 0;

cleanup:
    if (target_ide_idx != 0)
        unlock_resource(0, 0, 0, 0, target_ide_idx, 0);
    if (dst_ide_idx != 0)
        unlock_resource(0, 0, 0, 0, dst_ide_idx, 0);
    fclose(source_file);
    return ret_val;

cleanup_allocate1:
    if (dst_ide_idx != 0)
        unlock_resource(0, 0, 0, 0, dst_ide_idx, 0);
    fclose(source_file);
    return ENOSPC;

cleanup_allocate2:
    if (dst_ide_idx != 0)
        unlock_resource(0, 0, 0, 0, dst_ide_idx, 0);
    fclose(source_file);
    return ENOSPC;

cleanup_delete_inode1:
    delete_inode(target_ide, target_ide_idx);
    unlock_resource(0, 0, 0, 0, target_ide_idx, dst_parent_ide_idx);
    fclose(source_file);
    return ret_val;

cleanup_delete_inode2:
    delete_inode(target_ide, target_ide_idx);
    unlock_resource(0, 0, 0, 0, target_ide_idx, dst_ide_idx);
    fclose(source_file);
    return ret_val;
}