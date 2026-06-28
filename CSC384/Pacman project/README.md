# Pacman AI Agent

CSC384 — Introduction to Artificial Intelligence | University of Toronto Mississauga | Grade: 80/100

## Description
Implements intelligent agents for the Pacman game using classical search algorithms from UC Berkeley's CS188 project.

## Algorithms Implemented
- **DFS** — Depth-First Search
- **BFS** — Breadth-First Search  
- **UCS** — Uniform Cost Search
- **A\*** — A-Star Search with custom heuristics
- Greedy best-first search

## How to Run
```bash
cd search
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

## Key Files
- `search.py` — search algorithm implementations
- `searchAgents.py` — agent and heuristic definitions

## Tech Stack
Python, graph search, heuristic design
