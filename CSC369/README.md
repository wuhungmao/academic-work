# CSC369 — Operating Systems

University of Toronto Mississauga | Fall 2025

Operating systems assignments covering system calls, synchronization, virtual memory, and file systems — all implemented in C.

## Assignments

### A1 — System Calls
Custom system call implementation and kernel-space/user-space interaction.

### A2 — Synchronization
Concurrent programming with Pthreads: mutex locks, condition variables, semaphores. Solving classic synchronization problems (producer-consumer, readers-writers).

### A3 — Virtual Memory
Page table simulation, TLB management, page replacement policies (LRU, clock), and demand paging.

### A4 — Unix-like File System ⭐
Led a team to implement a complete Unix-like file system driver in C using `mmap` for direct disk image manipulation.

**Features:**
- Superblock, inode table, and bitmap management
- Hard links and symbolic links
- Full path resolution
- Thread-safe access with fine-grained mutex locks

## Tech Stack
C, Pthreads, mmap, POSIX APIs, GDB
