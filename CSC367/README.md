# CSC367 — Parallel Programming

University of Toronto Mississauga | Fall 2023

High-performance computing assignments covering memory architecture, parallelism with Pthreads, OpenMP, MPI, and CUDA.

## Projects

### Bandwidth and Memory Latency Testing
Benchmarks memory bandwidth and latency across cache levels (L1, L2, L3, DRAM) using pointer chasing and streaming patterns.

### Image Processing with Pthread
Parallel image processing pipeline using POSIX threads. Applies filters across PGM image rows/columns with a thread pool.

### Image Processing with GPU ⭐ 100/100
Processed a 10 MB × 10 MB PGM image in under 10 ms using CUDA — **70× speedup** over sequential. Optimized memory access patterns, tiled shared memory, and kernel launch configuration.

### Database Join with OpenMP & MPI
Parallelized relational database join operations:
- **OpenMP** for shared-memory multi-core parallelism
- **MPI** for distributed-memory multi-node execution
- Merged 10,000+ queries in under 200 ms

## Tech Stack
C, CUDA, Pthreads, OpenMP, MPI, Nsight Systems
