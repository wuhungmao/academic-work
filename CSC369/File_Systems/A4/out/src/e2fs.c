/*
 *------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC369H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * All of the files in this dir and all subdirectories are:
 * Copyright (c) 2025 MCS @ UTM
 * -------------
 */

/**
 * TODO: Make sure to add all necessary includes here
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "e2fs.h"

// additional header file below
#include "ext2.h"
#include "ext2fsal.h"
#include <errno.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <sys/stat.h>

/*
 * Helper function to get the next component of the path
 */
int get_path_component(const char *path, char *dir_name, int idx)
{
    assert(path[idx] == '/');
    idx++;
    int dir_name_len = 0;
    while (path[idx] != '/' && path[idx] != '\0')
    {
        dir_name[dir_name_len] = path[idx];
        dir_name_len++;
        idx++;
    }
    dir_name[dir_name_len] = '\0';
    return dir_name_len;
}

/*
 * Return 1 if success, otherwise return 0 on failure
 */
int compare_str(char entry_name[], unsigned char entry_name_len, char dir_name[], int dir_name_len)
{
    if (entry_name_len != dir_name_len)
    {
        return 0;
    }
    // use memcmp to increase efficiency and save lines
    return memcmp(entry_name, dir_name, entry_name_len) == 0;
}

/*
 * locks acquiring order: inode mutex1, inode mutex2, block bitmap, inode bitmap, group descriptor, super block
 * pass in 1 for locking the resource, 0 for doing nothing
 * Helper function to enforce a certain locking order to prevent deadlock
 * It also prevent locking and unlocking same inode twice
 */
void lock_resource(int lock_sb, int lock_gb, int lock_block_bitmap, int lock_inode_bitmap, unsigned int lock_inode_idx1, unsigned int lock_inode_idx2)
{
    if (lock_inode_idx1 > 0 && lock_inode_idx2 > 0)
    {
        if (lock_inode_idx1 < lock_inode_idx2)
        {
            pthread_mutex_lock(&inode_mutex[lock_inode_idx1 - 1]);
            pthread_mutex_lock(&inode_mutex[lock_inode_idx2 - 1]);
        }
        else if (lock_inode_idx1 > lock_inode_idx2)
        {
            pthread_mutex_lock(&inode_mutex[lock_inode_idx2 - 1]);
            pthread_mutex_lock(&inode_mutex[lock_inode_idx1 - 1]);
        }
        else
        {
            pthread_mutex_lock(&inode_mutex[lock_inode_idx1 - 1]);
        }
    }
    else
    {
        // Only one inode was requested
        if (lock_inode_idx1 > 0)
            pthread_mutex_lock(&inode_mutex[lock_inode_idx1 - 1]);
        if (lock_inode_idx2 > 0)
            pthread_mutex_lock(&inode_mutex[lock_inode_idx2 - 1]);
    }
    if (lock_sb)
        pthread_mutex_lock(&sb_mutex);
    if (lock_gb)
        pthread_mutex_lock(&group_desc_mutex);
    if (lock_block_bitmap)
        pthread_mutex_lock(&block_bitmap_mutex);
    if (lock_inode_bitmap)
        pthread_mutex_lock(&inode_bitmap_mutex);
}

/*
 * unlocks order: inode mutex1, inode mutex2, inode-bitmap,  block-bitmap, group descriptor, sb
 * pass in 1 for unlocking resource, 0 for doing nothing
 * need to unlock in reverse order to prevent lock inversion
 */
void unlock_resource(int unlock_sb, int unlock_gb, int unlock_block_bitmap, int unlock_inode_bitmap, unsigned int unlock_inode_idx1, unsigned int unlock_inode_idx2)
{
    if (unlock_inode_idx2 > 0)
        pthread_mutex_unlock(&inode_mutex[unlock_inode_idx2 - 1]);
    // to prevent some weird bug that two inode passed in is the same
    if (unlock_inode_idx1 > 0 && unlock_inode_idx1 != unlock_inode_idx2)
        pthread_mutex_unlock(&inode_mutex[unlock_inode_idx1 - 1]);
    if (unlock_inode_bitmap)
        pthread_mutex_unlock(&inode_bitmap_mutex);
    if (unlock_block_bitmap)
        pthread_mutex_unlock(&block_bitmap_mutex);
    if (unlock_gb)
        pthread_mutex_unlock(&group_desc_mutex);
    if (unlock_sb)
        pthread_mutex_unlock(&sb_mutex);
}

/*
 * search for a file/directory in a data block.
 * return 0 if not found, else return inode number
 */
unsigned int search_in_data_block(int data_block_idx, char *name, int name_len)
{
    // search through that data block to find the directory entry matching directory_name
    int offset = 0;
    while (offset < EXT2_BLOCK_SIZE)
    {
        // Find the memory location of that data block, then cast to a ext2_dir_entry
        struct ext2_dir_entry *dir_entry = (struct ext2_dir_entry *)(disk + EXT2_BLOCK_SIZE * data_block_idx + offset);
        // if we have deleted directory entry, its inode is set to 0, then compare_str would dir_entry->inode would return 0
        if (compare_str(dir_entry->name, dir_entry->name_len, name, name_len) && dir_entry->inode != 0)
        {
            // found it, move curr_inode to that inode
            return dir_entry->inode;
        }
        // help catching bug
        assert(dir_entry->rec_len != 0);
        offset += dir_entry->rec_len;
    }
    return 0;
}

/*
 * Given an inode with mode d(directory inode), find the inode number of the file/directory with name same as passed in name.
 * Return 0 if not found, otherwise return inode index
 */
unsigned int search_in_inode(struct ext2_inode *curr_inode, char *name, int name_len)
{
    assert((curr_inode->i_mode & 0xF000) == EXT2_S_IFDIR);
    unsigned int data_block_idx;
    for (int i = 0; i < 13; i++)
    {
        // handle indirect blocks
        if (i == 12)
        {
            data_block_idx = curr_inode->i_block[i];
            if (data_block_idx == 0)
            {
                continue;
            }
            unsigned int *indirect_blocks_ptr = (unsigned int *)(disk + EXT2_BLOCK_SIZE * data_block_idx);
            // block size divided by 32-bit pointers(4 bytes) = num of pointers in the block
            for (int j = 0; j < EXT2_BLOCK_SIZE / sizeof(unsigned int); j++)
            {
                int indirect_data_block_idx = indirect_blocks_ptr[j];
                if (indirect_data_block_idx == 0)
                {
                    continue;
                }
                unsigned int ide_idx = search_in_data_block(indirect_data_block_idx, name, name_len);
                if (ide_idx != 0)
                {
                    return ide_idx;
                }
            }
        }
        else
        {
            // handle direct blocks
            // get the data block pointed by the current directory
            data_block_idx = curr_inode->i_block[i];
            if (data_block_idx == 0)
            {
                continue;
            }
            unsigned int ide_idx = search_in_data_block(data_block_idx, name, name_len);
            if (ide_idx != 0)
            {
                return ide_idx;
            }
        }
    }
    return 0;
}

/*
 * Return inode at inode_idx
 */
struct ext2_inode *lookup_inode_by_idx(int inode_idx)
{
    // assume only one block group, which all inodes are stored in inode_table
    assert(inode_idx >= 1 && inode_idx <= INODE_COUNT);
    return &inode_table[inode_idx - 1];
}

/*
 * Returns the inode number to the last file/directory in the path
 * Returns positive inode number if the path/file exists
 * Returns 0 if the path/file does not exist at the last component
 * Returns -ENOENT if any component beside last one in the path does
 * not exist/exists as a file/exists as a symbolic link/
 * Returns -EIO if inode type is not identified
 * Returns -ENAMETOOLONG if name too long
 * Returns -1 on other errors
 */
int lookup_inode_by_path(const char *path, unsigned int *parent_node_idx, char *dir_name)
{
    int path_len = strlen(path);

    // name longer than maximum length
    if (path_len >= EXT2_NAME_LEN)
    {
        return -ENAMETOOLONG;
    }

    int idx = 0;
    int dir_name_len = 0;
    // remove trailing slashes
    if (path[path_len - 1] == '/' && path_len != 1)
    {
        while (path_len > 0 && path[path_len - 1] == '/')
        {
            path_len--;
        }
    }

    // start with inode that contain root directory, inode num 2, which is at index 1
    struct ext2_inode *curr_inode = lookup_inode_by_idx(2);
    int ide_idx = 2;
    int locked_id;
    // break down the path given into components
    // lock an inode, and unlock it when we find next inode in next path component
    while (idx < path_len)
    {
        locked_id = ide_idx;
        lock_resource(0, 0, 0, 0, locked_id, 0);
        assert(path[idx] == '/');
        // dir_name contains the name of next component, dir_name_len is its length
        dir_name_len = get_path_component(path, dir_name, idx);

        if (idx + dir_name_len + 1 == path_len)
        {
            // this is the last component, return the inode number here
            if ((curr_inode->i_mode & 0xF000) == EXT2_S_IFDIR)
            {
                *parent_node_idx = ide_idx;
                ide_idx = search_in_inode(curr_inode, dir_name, dir_name_len);
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                if (ide_idx != 0)
                {
                    return ide_idx;
                }
                else
                {
                    return 0; // unable to find file/directory
                }
            }
            else if ((curr_inode->i_mode & 0xF000) == EXT2_S_IFREG)
            {
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                return -ENOENT; // this inode is a file
            }
            else if ((curr_inode->i_mode & 0xF000) == EXT2_S_IFLNK)
            {
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                return -ENOENT; // this inode is a symbolic link
            }
            else
            {
                perror("Unknown inode type");
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                return -EIO; // does not exist
            }
        }
        else
        {
            // not the last component, just traverse
            if ((curr_inode->i_mode & 0xF000) == EXT2_S_IFDIR)
            {
                ide_idx = search_in_inode(curr_inode, dir_name, dir_name_len);
                // unlock currently holding lock
                unlock_resource(0, 0, 0, 0, locked_id, 0);

                if (ide_idx != 0)
                {
                    // found it, move curr_inode to that inode
                    curr_inode = lookup_inode_by_idx(ide_idx);
                    idx += dir_name_len + 1;
                }
                else
                {
                    return -ENOENT; // unable to find directory in the path
                }
            }
            else if ((curr_inode->i_mode & 0xF000) == EXT2_S_IFREG)
            {
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                return -ENOENT; // this inode is a file
            }
            else if ((curr_inode->i_mode & 0xF000) == EXT2_S_IFLNK)
            {
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                return -ENOENT; // this inode is a symbolic link
            }
            else
            {
                perror("Unknown inode type");
                unlock_resource(0, 0, 0, 0, locked_id, 0);
                return -EIO; // does not exist
            }
        }
    }
    return 0;
}

/*
 * set inode fields
 * Return 0 on success
 * Return ENOSPC, EIO if failure
 */
unsigned int set_inode_fields(struct ext2_inode *inode, unsigned short i_mode)
{
    inode->i_mode = i_mode;

    // iniitialize all block to 0
    for (int i = 0; i < 15; i++)
    {
        inode->i_block[i] = 0;
    }

    if ((inode->i_mode & 0xF000) == EXT2_S_IFDIR)
    {
        // directory, size is one data block
        inode->i_size = EXT2_BLOCK_SIZE;
        inode->i_blocks = EXT2_BLOCK_SIZE / 512;
        // We need to allocate one data block for the directory inode to hold . and ..
        int free_data_block_idx = allocate_data_block();
        CHECK_ERROR_INDEX(free_data_block_idx);
        inode->i_block[0] = free_data_block_idx;

        // set link count to 2. One from ., another one come from parent directory
        inode->i_links_count = 2;
    }
    else if ((inode->i_mode & 0xF000) == EXT2_S_IFREG)
    {
        // regular file, empty size
        inode->i_size = 0;

        // No data block allocated
        inode->i_blocks = 0;

        // Parent directory hold one directory entry to the file
        inode->i_links_count = 1;
    }
    else if ((inode->i_mode & 0xF000) == EXT2_S_IFLNK)
    {
        // Symbolic link, need to set it to the size of target length
        inode->i_size = 0;

        // No data blocks allocated yet
        inode->i_blocks = 0;

        // Set link count
        inode->i_links_count = 1;
    }
    else
    {
        perror("Unknown inode type");
        return EIO;
    }

    // all 0 for following fields
    inode->osd1 = 0;
    inode->i_generation = 0;
    inode->i_file_acl = 0;
    inode->i_dir_acl = 0;
    inode->i_faddr = 0;
    inode->extra[0] = 0;
    inode->extra[1] = 0;
    inode->extra[2] = 0;
    inode->i_uid = 0;
    // prof said we don't need to worry about time fields except i_dtime
    inode->i_ctime = 0;
    inode->i_dtime = 0;
    inode->i_atime = 0;
    inode->i_mtime = 0;
    inode->i_gid = 0;
    return 0;
}

/*
 * Allocate a new inode and return its inode number
 * Returns 0 on failure, otherwise return index
 * Updates the inode bitmap accordingly
 * Updates the superblock and group descriptor accordingly
 * initializing the inode fields
 * Locks superblock, group descriptor, inode bitmap
 * The caller is responsible for locking the inode mutex before allocation
 */
unsigned int allocate_inode(unsigned short i_mode)
{
    // lock super block, group descriptor, inode bitmap
    lock_resource(1, 1, 0, 1, 0, 0);

    if (sb->s_free_inodes_count <= 0)
    {
        // no free inode left
        unlock_resource(1, 1, 0, 1, 0, 0);
        return 0;
    }

    int free_inode_idx = -1;
    // we have 32 inode, so print 32 / 8 = 4 bytes from the bitmap. Each bit represents a inode.
    for (int i = 0; i < sb->s_inodes_count; i++)
    {
        if ((i + 1 > 11) && !(inode_bitmap[i / 8] & (1 << (i % 8))))
        {
            // found the next free inode
            free_inode_idx = i + 1; // inode is 1-index based

            // mark that inode as used
            inode_bitmap[i / 8] |= (1 << (i % 8));

            // reduce number of free inodes for both superblock and group descriptor
            sb->s_free_inodes_count -= 1;
            block_group_desc->bg_free_inodes_count -= 1;

            break;
        }
    }

    if (free_inode_idx == -1)
    {
        // cannot find a free inode
        unlock_resource(1, 1, 0, 1, 0, 0);
        return 0;
    }
    unlock_resource(1, 1, 0, 1, 0, 0);

    lock_resource(0, 0, 0, 0, free_inode_idx, 0);
    struct ext2_inode *allocated_inode = lookup_inode_by_idx(free_inode_idx);
    // initialize the inode fields to 0
    memset(allocated_inode, 0, sizeof(struct ext2_inode));

    set_inode_fields(allocated_inode, i_mode);
    unlock_resource(0, 0, 0, 0, free_inode_idx, 0);
    return free_inode_idx; // inode number is 1-based
}

/*
 * Returns the index of the next free data block or 0 on failure
 * Updates the block bitmap accordingly
 * Memset a data block to 0
 * Locks superblock, group descriptor, block bitmap
 */
unsigned int allocate_data_block()
{
    lock_resource(1, 1, 1, 0, 0, 0);
    if (sb->s_free_blocks_count <= 0)
    {
        // no more free blocks
        unlock_resource(1, 1, 1, 0, 0, 0);
        return 0;
    }

    // Find next free data block
    int i, ii, is_found = 0;
    for (i = 0; i < sb->s_blocks_count / 8; ++i)
    {
        unsigned char byte = (unsigned char)block_bitmap[i];
        for (ii = 0; ii < 8; ++ii)
        {
            if ((byte & (1 << ii)) == 0)
            { // Found a free block
                is_found = 1;
                break;
            }
        }
        if (is_found)
        {
            break;
        }
    }
    if (is_found == 0)
    {
        unlock_resource(1, 1, 1, 0, 0, 0);
        return 0;
    }

    // Mark that block as used
    block_bitmap[i] = block_bitmap[i] | (1 << ii);

    // Update superblock
    sb->s_free_blocks_count -= 1;

    // Update group descriptor
    block_group_desc->bg_free_blocks_count -= 1;

    unsigned char *data_block = (unsigned char *)(disk + EXT2_BLOCK_SIZE * ((i * 8) + ii));

    // memset data block to 0
    memset(data_block, 0, EXT2_BLOCK_SIZE);
    unlock_resource(1, 1, 1, 0, 0, 0);

    return (i * 8) + ii;
}

/*
 * Finds space after the last entry in a directory data block that can store a new entry of <min_size> length.
 * Reserve the space and sets the rec_len for the previous and new entry (if the entries exist).
 * Returns the offset into the data block, or EXT2_BLOCK_SIZE if there is insufficient space.
 */
int make_space_for_dir_entry(unsigned int dir_block_i, unsigned short min_size)
{
    unsigned char *dir_block = (unsigned char *)(disk + EXT2_BLOCK_SIZE * dir_block_i);
    int offset = 0;
    struct ext2_dir_entry *last_dir_entry = (struct ext2_dir_entry *)(dir_block + offset);

    // data block is just allocated and memset to 0, need to initialize it and return offset 0
    if (last_dir_entry->rec_len == 0)
    {
        last_dir_entry->rec_len = EXT2_BLOCK_SIZE;
        return 0;
    }

    while (offset + last_dir_entry->rec_len < EXT2_BLOCK_SIZE)
    {
        // add check here to ensure we don't fall into infinite loop
        if (last_dir_entry->rec_len == 0)
        {
            return EXT2_BLOCK_SIZE;
        }

        offset += last_dir_entry->rec_len;
        last_dir_entry = (struct ext2_dir_entry *)(dir_block + offset);
    }

    unsigned short last_size = sizeof(struct ext2_dir_entry) + last_dir_entry->name_len;
    last_size = ((last_size + 3) / 4) * 4; // Align to 4 bytes

    // Safety check to prevent underflow debug printing
    if (last_dir_entry->rec_len < last_size)
    {
        return EXT2_BLOCK_SIZE;
    }

    unsigned short remaining_space = last_dir_entry->rec_len - last_size;

    if (remaining_space >= min_size)
    { // There is enough space in padding of last entry
        last_dir_entry->rec_len = last_size;
        offset += last_size;
        // ensure we are not returning same address
        assert(offset != 0);
        last_dir_entry = (struct ext2_dir_entry *)(dir_block + offset);
        last_dir_entry->rec_len = remaining_space;

        return offset;
    }

    return EXT2_BLOCK_SIZE;
}

/*
 * Create an entry for the directory at inode <dir_inode_num>.
 * Return 1 on success, ENOSPC on failure
 * Already acquire lock at the caller side
 */
int create_dir_entry(unsigned int dir_inode_num, char *name, unsigned int inode_num, unsigned char file_type)
{
    // Calculate ext2_dir_entry metadata
    struct ext2_inode *dir_inode = lookup_inode_by_idx(dir_inode_num);
    int name_len = strnlen(name, EXT2_NAME_LEN);
    unsigned short min_rec_len = sizeof(struct ext2_dir_entry) + name_len;
    min_rec_len = ((min_rec_len + 3) / 4) * 4; // Align to 4 bytes

    int num_blocks = dir_inode->i_blocks / 2; // 1 data block = 2 disk sectors

    unsigned int dir_block_i; // Index of block to add entry to
    int offset = 0;           // Offset of new entry in block

    if (num_blocks == 0)
    { // No blocks allocated (empty directory)
        // Allocate new data block
        dir_block_i = allocate_data_block();
        CHECK_ERROR_INDEX(dir_block_i);

        dir_inode->i_block[0] = dir_block_i;
        dir_inode->i_blocks += 2; // +2 disk sectors
        dir_inode->i_size += EXT2_BLOCK_SIZE;
    }
    else if (num_blocks <= 12)
    { // The last block allocated is a direct block
        dir_block_i = dir_inode->i_block[num_blocks - 1];
        offset = make_space_for_dir_entry(dir_block_i, min_rec_len);

        if (offset >= EXT2_BLOCK_SIZE)
        { // Not enough space in last block

            // Allocate new data block
            dir_block_i = allocate_data_block();
            CHECK_ERROR_INDEX(dir_block_i);
            dir_inode->i_block[num_blocks] = dir_block_i;
            dir_inode->i_blocks += 2; // +2 disk sectors
            dir_inode->i_size += EXT2_BLOCK_SIZE;

            if (num_blocks == 12)
            { // We allocated an indirect block

                unsigned char *indir_block = (unsigned char *)(disk + EXT2_BLOCK_SIZE * dir_inode->i_block[num_blocks]);
                // Allocate new data block again, to be the first block in the indirect block
                dir_block_i = allocate_data_block();
                CHECK_ERROR_INDEX(dir_block_i);
                memcpy(indir_block, &dir_block_i, sizeof(unsigned int));
                dir_inode->i_blocks += 2; // +2 disk sectors
                dir_inode->i_size += EXT2_BLOCK_SIZE;
            }
            offset = 0;
            struct ext2_dir_entry *first_entry = (struct ext2_dir_entry *)(disk + EXT2_BLOCK_SIZE * dir_block_i);
            first_entry->rec_len = EXT2_BLOCK_SIZE;
        }
    }
    else
    { // Last block is in the single indirect block (double/triple are unused for the assignment)
        unsigned char *indir_block = (unsigned char *)(disk + EXT2_BLOCK_SIZE * dir_inode->i_block[12]);
        num_blocks -= 13; // Blocks pointed by indirect block (-12 direct blocks, -1 indirect block)
        dir_block_i = ((unsigned int *)indir_block)[num_blocks - 1];

        offset = make_space_for_dir_entry(dir_block_i, min_rec_len);
        if (offset >= EXT2_BLOCK_SIZE)
        { // Not enough space in last block

            if (num_blocks == EXT2_BLOCK_SIZE / sizeof(unsigned int))
            { // indirect block is full
                return ENOSPC;
            }
            // Allocate a new block
            unsigned char *indir_block = (unsigned char *)(disk + EXT2_BLOCK_SIZE * dir_inode->i_block[12]);
            dir_block_i = allocate_data_block();
            CHECK_ERROR_INDEX(dir_block_i);
            memcpy(&(indir_block[num_blocks * sizeof(unsigned int)]), &dir_block_i, sizeof(unsigned int));
            dir_inode->i_blocks += 2; // +2 disk sectors
            dir_inode->i_size += EXT2_BLOCK_SIZE;
            offset = make_space_for_dir_entry(dir_block_i, min_rec_len);
        }
    }

    struct ext2_dir_entry *new_dir_entry = (struct ext2_dir_entry *)(disk + EXT2_BLOCK_SIZE * dir_block_i + offset);
    // Write the directory entry into the data block
    new_dir_entry->inode = inode_num;
    // new_dir_entry->rec_len is set by the make_space_for_entry(...) call
    new_dir_entry->name_len = name_len;
    new_dir_entry->file_type = file_type;
    // ensure directory entry name is not null terminated by not copying null terminator
    strncpy(new_dir_entry->name, name, name_len);

    return 1;
}

/*
 * delete a directory entry in a data block given data block index,
 * entry name, and size of entry name
 * return 1 upon success, 0 otherwise
 */
int delete_dir_entry_in_data_block(int data_block_idx, char *entry_name, int entry_name_len)
{
    int offset = 0;
    // remove that directory entry and update rec_len of previous directory entry
    struct ext2_dir_entry *prev_dir_entry = NULL;
    struct ext2_dir_entry *curr_dir_entry = NULL;
    while (offset < EXT2_BLOCK_SIZE)
    {
        // Find the memory location of that data block, then cast to a ext2_dir_entry
        curr_dir_entry = (struct ext2_dir_entry *)(disk + EXT2_BLOCK_SIZE * data_block_idx + offset);

        // Safety check for corrupted blocks
        if (curr_dir_entry->rec_len == 0)
        {
            return 0;
        }

        if (curr_dir_entry->inode != 0 && compare_str(curr_dir_entry->name, curr_dir_entry->name_len, entry_name, entry_name_len))
        {
            // Found the directory entry we want to delete
            // Update rec len of previous directory entry to remove current directory entry
            if (prev_dir_entry == NULL)
            {
                //  the first entry match, set inode to 0 like what we saw in lab
                curr_dir_entry->inode = 0;
            }
            else
            {
                prev_dir_entry->rec_len += curr_dir_entry->rec_len;
            }
            return 1;
        }
        offset += curr_dir_entry->rec_len;
        assert(offset != 0);
        prev_dir_entry = curr_dir_entry;
    }
    // didn't find any directory entry matched with directory name
    return 0;
}

/*
 * In the directory at inode dir_inode_num, delete the file with name
 * Does not support deleting directories (See assignment handout)
 * return 1 upon successful, ENOENT upon failure
 */
int delete_dir_entry(int dir_inode_num, char *name)
{
    struct ext2_inode *parent_ide = lookup_inode_by_idx(dir_inode_num);

    // search through ext2_inode.i_block[0-12], 0-11 are direct, 12 are single indirect, 13 and 14 are unused for this assignment
    int data_block_idx;
    for (int i = 0; i < 13; i++)
    {
        // handle indirect blocks
        if (i == 12)
        {
            data_block_idx = parent_ide->i_block[i];
            if (data_block_idx == 0)
            {
                continue;
            }
            unsigned int *indirect_blocks_ptr = (unsigned int *)(disk + EXT2_BLOCK_SIZE * data_block_idx);
            // block size divided by 32-bit pointers(4 bytes) = num of pointers in the block
            for (int j = 0; j < EXT2_BLOCK_SIZE / sizeof(unsigned int); j++)
            {
                int indirect_data_block_idx = indirect_blocks_ptr[j];
                if (indirect_data_block_idx == 0)
                {
                    continue;
                }

                if (delete_dir_entry_in_data_block(indirect_data_block_idx, name, strlen(name)))
                {
                    return 1;
                }
            }
        }
        else
        {
            // handle direct blocks
            // get the data block pointed by the current directory
            data_block_idx = parent_ide->i_block[i];
            if (data_block_idx == 0)
            {
                continue;
            }

            if (delete_dir_entry_in_data_block(data_block_idx, name, strlen(name)) == 1)
            {
                return 1;
            }
        }
    }
    return ENOENT;
}

/*
 * Unset block bitmap to free all data blocks currently held by inode
 */
void unset_block_bitmap(struct ext2_inode *inode)
{
    lock_resource(0, 0, 1, 0, 0, 0);
    int data_block_idx;
    for (int i = 0; i < 13; i++)
    {
        // handle indirect blocks
        if (i == 12)
        {
            data_block_idx = inode->i_block[i];
            if (data_block_idx == 0)
            {
                continue;
            }

            unsigned int *indirect_blocks_ptr = (unsigned int *)(disk + EXT2_BLOCK_SIZE * data_block_idx);
            unsigned char *byte;
            // block size divided by 32-bit pointers(4 bytes) = num of pointers in the block
            for (int j = 0; j < EXT2_BLOCK_SIZE / sizeof(unsigned int); j++)
            {
                int indirect_data_block_idx = indirect_blocks_ptr[j];
                if (indirect_data_block_idx == 0)
                {
                    continue;
                }
                else
                {
                    // set bit at indiret_data_block_idx as 0
                    byte = &block_bitmap[indirect_data_block_idx / 8];
                    int bit_idx = (indirect_data_block_idx % 8);
                    int mask = ~(1 << bit_idx);
                    *byte = *byte & mask;
                    indirect_blocks_ptr[j] = 0;
                }
            }

            // free the indirect table data block itself
            byte = &block_bitmap[data_block_idx / 8];
            int bit_idx = (data_block_idx % 8);
            int mask = ~(1 << bit_idx);
            *byte = *byte & mask;

            inode->i_block[i] = 0;
        }
        else
        {
            // handle direct blocks
            // get the data block pointed by the current directory
            data_block_idx = inode->i_block[i];
            if (data_block_idx == 0)
            {
                continue;
            }
            else
            {

                unsigned char *byte = &block_bitmap[data_block_idx / 8];
                int bit_idx = (data_block_idx % 8);
                int mask = ~(1 << bit_idx);
                *byte = *byte & mask;
                inode->i_block[i] = 0;
            }
        }
    }
    unlock_resource(0, 0, 1, 0, 0, 0);
}

/*
 * Delete inode
 * This including changing superblock, block group descriptor, update inode_bitmap, unset block bitmap
 * return nothing, a void function
 * inode is locked by caller
 */
void delete_inode(struct ext2_inode *inode, int ide_idx)
{
    lock_resource(1, 1, 0, 1, 0, 0);
    inode->i_dtime = (unsigned int)time(NULL);

    // increment super block free inodes count
    sb->s_free_inodes_count++;
    sb->s_free_blocks_count += (inode->i_blocks / 2);

    // increment bg_free_blocks_count by number of data block
    block_group_desc->bg_free_blocks_count += (inode->i_blocks / 2);
    block_group_desc->bg_free_inodes_count++;

    // unset the bit at ide_idx in inode_bitmap
    int byte_idx = (ide_idx - 1) / 8;
    int bit_idx = (ide_idx - 1) % 8;
    unsigned char *byte = &inode_bitmap[byte_idx];

    // create the mask to unset the bit at bit_idx for byte at byte_idx
    int mask = ~(1 << bit_idx);
    *byte = *byte & mask;

    // unset the bit at data block idx in block bitmap
    unset_block_bitmap(inode);
    unlock_resource(1, 1, 0, 1, 0, 0);
}

/* Given a file pointer, return the size of a file
 * used by cp to determine the number of bytes of this source file
 * Return size if succeed, EIO if fail
 */
long find_file_size(FILE *source_file)
{
    struct stat source_file_stat;
    int source_file_fd = fileno(source_file);
    if (fstat(source_file_fd, &source_file_stat) == 0)
    {
        return source_file_stat.st_size;
    }
    else
    {
        return -EIO;
    }
}

/*
 * This function copy data block from source file to target inode
 * Return 1 on success, ENOSPC on failure
 */
int cp_from_source_file(struct ext2_inode *target_ide, int target_ide_idx, FILE *source_file)
{
    // Need to copy data from source file to this inode, do it block by block
    int data_block_idx = 0;
    unsigned int free_data_block_idx;
    unsigned int indirect_table_idx;
    unsigned int *indirect_table;

    long src_file_size = find_file_size(source_file);

    // EIO
    if (src_file_size < 0)
    {
        return -src_file_size;
    }
    // empty source file, do nothing
    if (src_file_size == 0)
    {
        return 1;
    }

    // This is used to hold 1024 bytes from source file, to be copied to a data block
    char data_block_buffer[EXT2_BLOCK_SIZE];

    // determine how many data blocks to be allocated
    int data_blocks_num = (src_file_size + EXT2_BLOCK_SIZE - 1) / EXT2_BLOCK_SIZE;

    while (data_block_idx < data_blocks_num)
    {
        // allocate a data block
        free_data_block_idx = allocate_data_block();
        CHECK_ERROR_INDEX(free_data_block_idx);

        // Need to clear data from previous copy
        memset(data_block_buffer, 0, EXT2_BLOCK_SIZE);
        // need to read 1024 of 1 byte into the buffer
        fread(data_block_buffer, 1, EXT2_BLOCK_SIZE, source_file);

        // find the correct memory location on disk
        char *data_block_ptr = (char *)(disk + EXT2_BLOCK_SIZE * free_data_block_idx);
        memcpy(data_block_ptr, data_block_buffer, EXT2_BLOCK_SIZE);

        // assign data block to target inode
        if (data_block_idx < 12)
        {
            target_ide->i_block[data_block_idx] = free_data_block_idx;
            target_ide->i_blocks += 2; // Add 2 sectors (1024 bytes)
        }
        else
        {
            // TA guarantee we don't need to assign double indrect table and triple indirect table
            if (data_block_idx == 12)
            {
                indirect_table_idx = allocate_data_block();
                CHECK_ERROR_INDEX(indirect_table_idx);

                target_ide->i_block[12] = indirect_table_idx;
                target_ide->i_blocks += 2;
            }
            // Get pointer to the indirect table
            indirect_table = (unsigned int *)(disk + EXT2_BLOCK_SIZE * target_ide->i_block[12]);

            // The index inside the table is (i - 12)
            indirect_table[data_block_idx - 12] = free_data_block_idx;

            target_ide->i_blocks += 2;
        }
        data_block_idx++;
    }

    target_ide->i_size = src_file_size;
    target_ide->i_ctime = (unsigned int)time(NULL);
    target_ide->i_mtime = (unsigned int)time(NULL);

    return 1;
}

/*
 * Used by cp to update an inode when copying data to an existing inode
 * Return nothing, void function
 */
void clear_data_block(struct ext2_inode *target_ide, int target_ide_idx)
{
    lock_resource(1, 1, 0, 0, 0, 0);

    target_ide->i_size = 0;
    int i_blocks = target_ide->i_blocks;
    target_ide->i_blocks = 0;
    target_ide->i_mtime = time(NULL);

    // free data block pointed by inode and update i_dtime
    sb->s_free_blocks_count += (i_blocks / 2);

    // increment bg_free_blocks_count by number of data block
    block_group_desc->bg_free_blocks_count += (i_blocks / 2);

    // unset the bit at data block idx in block bitmap
    unset_block_bitmap(target_ide);

    // now data block is freed
    unlock_resource(1, 1, 0, 0, 0, 0);
}
