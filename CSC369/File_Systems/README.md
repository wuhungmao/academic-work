# Unix-like File System Simulation

CSC369 — Operating Systems | University of Toronto Mississauga | Fall 2025

## Description
A complete Unix-like file system driver implemented in C. Uses `mmap` to directly manipulate a binary disk image, implementing all major VFS layer operations.

## Features
- Superblock, free block bitmap, and inode table
- File creation, deletion, reading, and writing
- Directory operations (mkdir, rmdir, ls)
- Hard links and symbolic links
- Full path resolution (absolute and relative)
- Thread-safe access with fine-grained `pthread_mutex_t` per inode

## How to Build & Run
```bash
cd A4
make
./fs_sim <disk_image>
```

## Key Concepts
`mmap`, inode-based storage, bitmap allocation, path traversal, POSIX threading

## Tech Stack
C, mmap, Pthreads, POSIX file I/O
