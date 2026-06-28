# Image Processing with GPU

CSC367 — Parallel Programming | University of Toronto Mississauga | Grade: 100/100

## Description
Parallel image processing on large PGM images using CUDA. Achieves a **70× speedup** over sequential C, processing a 10 MB × 10 MB image in under 10 ms.

## Optimizations
- Tiled shared memory to reduce global memory bandwidth
- Coalesced memory access patterns
- Tuned thread block dimensions for SM occupancy
- Minimized host-device transfers

## How to Build & Run
```bash
cmake -S . -B build
cmake --build build
./build/main <input.pgm> <output.pgm>
```

See the starter README in `googletest/` for PGM image creation utilities.

## Tech Stack
C, CUDA, CMake, Google Test
