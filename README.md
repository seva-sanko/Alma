# Parallel Programming on AlmaLinux

Research work on parallel programming technologies in Linux-based operating systems (AlmaLinux).

## Goal

Design a computer network and develop a terminal application implementing graph theory algorithms using parallel programming — OpenMP (shared memory) and OpenMPI (distributed memory).

## Environment

- VirtualBox virtual machines running **AlmaLinux**
- Network: `10.0.60.0/24`
- Language: C++

## Algorithms implemented

| Algorithm | Description |
|-----------|-------------|
| Kirchhoff's matrix-tree theorem | Count the number of spanning trees |
| Prim's algorithm | Minimum spanning tree |
| Prüfer sequence | Tree encoding/decoding |
| Graph coloring | Minimum vertex coloring |

All algorithms have parallel versions using **OpenMP** and/or **OpenMPI**.

## Contents

- `lost.cpp` — C++ implementation of the graph algorithms
- `Sanko.tex` — LaTeX source of the research report (62 pages)
- `Sanko.pdf` — compiled report
- `img/` — screenshots and diagrams used in the report
