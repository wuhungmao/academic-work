# Futoshiki Puzzle Solver

CSC384 — Introduction to Artificial Intelligence | University of Toronto Mississauga

## Description
Solves Futoshiki inequality puzzles using Constraint Satisfaction Problem (CSP) techniques.

Futoshiki is a Latin square puzzle with inequality constraints between adjacent cells. The solver uses backtracking search enhanced with constraint propagation.

## Techniques
- Backtracking search with forward checking
- Arc consistency (AC-3) for constraint propagation
- MRV (Minimum Remaining Values) heuristic
- Degree heuristic for variable ordering

## How to Run
```bash
python3 futoshiki.py <puzzle_file>
```

## Tech Stack
Python, CSP, backtracking, arc consistency
