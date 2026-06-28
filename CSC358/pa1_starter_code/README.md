# Ethernet Frame & ARP Request Simulation

CSC358 — Computer Networks | University of Toronto Mississauga | Feb 2024

## Description
Simulates how network switches handle Ethernet frames and ARP (Address Resolution Protocol) requests in C++. Based on Stanford's CS 144 networking labs.

Implements a `NetworkInterface` that maps IP addresses to MAC addresses, sending ARP requests to resolve unknown addresses and caching replies.

## Key Files
- `src/network_interface.cc` — main implementation
- `src/network_interface.hh` — interface definition

## Key Concepts
- Ethernet frame encapsulation (Layer 2)
- ARP request/reply protocol
- IP-to-MAC address resolution and caching
- Switch forwarding tables

## How to Build & Run
```bash
cmake -S . -B build
cmake --build build
cmake --build build --target test   # run tests
```

## Tech Stack
C++, CMake, Ethernet/ARP protocols, Layer 2 networking
