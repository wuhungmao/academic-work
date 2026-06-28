# Bandwidth and Memory Latency Testing

CSC367 — Parallel Programming | University of Toronto Mississauga

## Description
Measures memory bandwidth and access latency across the cache hierarchy (L1, L2, L3, DRAM) using two standard microbenchmark techniques:

- **Part 1 — Bandwidth:** Streaming read/write benchmark to saturate memory bandwidth at each level
- **Part 2 — Latency:** Pointer-chasing benchmark to measure round-trip latency, defeating hardware prefetchers

## How to Run
```bash
cd part1 && make && ./bandwidth
cd part2 && make && ./latency
```

## Key Concepts
Cache hierarchy, memory access patterns, hardware prefetching, NUMA effects

## Tech Stack
C, x86 intrinsics, perf
