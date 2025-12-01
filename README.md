
# Coursework: 2 Adversarial Search (m,n,k Game)

This code allows the user to play and experiment with generalised tic-tac-toe games of size $m \times n$, where $k$ in a row wins. It supports:

- **Interactive gameplay** against an AI using minimax.
- **Automated test cases** to ensure correctness.
- **Performance experiments** to analyse the effect of pruning and board size.


## Usage

### 1. Play interactively

Play a game against the AI. The AI will recommend moves and play as 'O'.

```bash
python main.py --play \
    # --m: Number of rows for the board (default 4)
    # --n: Number of columns for the board (default 4)
    # --k: Number of consecutive symbols needed to win (default 4)
    # --prune: Use alpha-beta pruning for the AI (optional; speeds up decision-making)
````

**Example:**

```bash
python main.py --play --m 3 --n 3 --k 3 --prune
```

  * `--play` triggers the interactive game.
  * `--prune` enables alpha-beta pruning in minimax to reduce the number of nodes visited.

### 2\. Run automated tests

All test cases run automatically if you use `--play`. They check:

  * Minimax returns correct values on empty boards.
  * AI can detect winning moves for 'X'.
  * AI can block opponent moves ('O').
  * Correct handling when all moves are equally bad.

### 3\. Run timing experiments

Measure execution time and nodes visited for different board sizes and pruning options.

```bash
python main.py --experiments
```

  * `--experiments` runs predefined configurations and saves results to a CSV (`minimax_timings.csv`) for analysis.

**Default experiment configurations:**

  * $3 \times 3 \times 3$ with and without pruning
  * $3 \times 4 \times 3$ with and without pruning
  * $4 \times 3 \times 3$ with and without pruning
  * $4 \times 4 \times 3$ (pruning only)

> It is possible to edit the `configs` list in the script to test other board sizes.

-----

## Notes

  * The AI uses minimax and optionally alpha-beta pruning (`--prune`) to reduce nodes visited.
  * `nodes_visited` tracks the number of game tree nodes evaluated.
  * Experiment results are saved as CSV for further analysis.

<!-- end list -->