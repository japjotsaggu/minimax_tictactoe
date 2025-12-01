import argparse
import time
import pandas as pd

nodes_visited = 0  # Global counter for visited nodes during minimax


class Game:
    """
    Class to represent an (m,n,k) tic-tac-toe game with optional alpha-beta pruning.
    """

    def __init__(self, m, n, k, perform_prune=False):
        """
        Initialize the game parameters.
        m, n: board dimensions
        k: number of symbols in a row needed to win
        perform_prune: whether to use alpha-beta pruning in minimax
        """
        self.m = m
        self.n = n
        self.k = k
        self.board = []
        self.available_cells = set()
        self.perform_prune = perform_prune

    def initialize_game(self):
        """Initialize an empty board and the set of available cells."""
        self.board = [[""] * self.n for _ in range(self.m)]
        self.available_cells = {(r, c) for r in range(self.m) for c in range(self.n)}

    def drawboard(self):
        """Print the current board state in a readable format."""
        print("   " + "   ".join(str(i) for i in range(self.n)))
        for i, row in enumerate(self.board):
            print(f"{i}  " + " | ".join(cell if cell else " " for cell in row))
            if i < self.m - 1:
                print("   " + "-" * (self.n * 4 - 3))

    def is_terminal(self):
        """Check if the game has ended either by win or draw."""
        if self.check_winner() is not None:
            return True
        return len(self.available_cells) == 0

    def is_valid(self, move):
        """Check if a move is valid (i.e., the cell is empty)."""
        return move in self.available_cells

    def evaluate_game(self):
        """Return 1 if X wins, -1 if O wins, 0 otherwise."""
        winner = self.check_winner()
        if winner == "X":
            return 1
        elif winner == "O":
            return -1
        return 0

    def play(self):
        """Interactive game loop between human (X) and AI (O)."""
        while not self.is_terminal():
            self.drawboard()

            # Recommend moves for X
            _, recommended_moves = self.max(return_moves=True)
            print(f"Recommended moves: {recommended_moves}")

            # Human input
            players_move = self.get_move(input("Please enter your chosen move (row,column): "))
            row, col = players_move
            self.board[row][col] = "X"
            self.available_cells.remove((row, col))

            if self.is_terminal():
                break

            # AI move
            _, min_move = self.min(return_move=True)
            print(f"Min (O) plays: {min_move}")
            r, c = min_move
            self.board[r][c] = "O"
            self.available_cells.remove(min_move)

        # Game over
        self.drawboard()
        winner = self.check_winner()
        if winner is None:
            print("Game over: DRAW")
        else:
            print(f"Game over: {winner} wins")

    def get_move(self, move):
        """Read and validate a move from the user input."""
        while True:
            try:
                row, col = map(int, move.split(","))
                if self.is_valid((row, col)):
                    return (row, col)
                else:
                    move = input("Invalid move. Please enter a valid move (row,column): ")
            except Exception:
                move = input("Invalid format. Please enter your move as row,column: ")

    def max(self, alpha=float('-inf'), beta=float('inf'), return_moves=False):
        """
        Minimax max function for X.
        Returns best value or tuple (value, moves) if return_moves=True.
        Uses alpha-beta pruning if self.perform_prune=True.
        """
        global nodes_visited
        nodes_visited += 1

        if self.is_terminal():
            return (self.evaluate_game(), []) if return_moves else self.evaluate_game()

        best_value = float("-inf")
        best_moves = []

        for move in list(self.available_cells):
            r, c = move
            self.board[r][c] = "X"
            self.available_cells.remove(move)

            value = self.min(alpha, beta) if self.perform_prune else self.min()

            self.board[r][c] = ""
            self.available_cells.add(move)

            if value > best_value:
                best_value = value
                best_moves = [move] if return_moves else []
            elif return_moves and value == best_value:
                best_moves.append(move)

            # Alpha-beta pruning
            if self.perform_prune:
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break

        return (best_value, best_moves) if return_moves else best_value

    def min(self, alpha=float('-inf'), beta=float('inf'), return_move=False):
        """
        Minimax min function for O.
        Returns best value or tuple (value, move) if return_move=True.
        Uses alpha-beta pruning if self.perform_prune=True.
        """
        global nodes_visited
        nodes_visited += 1

        if self.is_terminal():
            return (self.evaluate_game(), None) if return_move else self.evaluate_game()

        best_value = float("inf")
        best_move = None

        for move in list(self.available_cells):
            r, c = move
            self.board[r][c] = "O"
            self.available_cells.remove(move)

            value = self.max(alpha, beta) if self.perform_prune else self.max()

            self.board[r][c] = ""
            self.available_cells.add(move)

            if value < best_value:
                best_value = value
                best_move = move if return_move else None

            # Alpha-beta pruning
            if self.perform_prune:
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return (best_value, best_move) if return_move else best_value

    def check_winner(self):
        """Check the board for a winner. Returns 'X', 'O', or None."""

        def check_direction(r, c, dr, dc, player, remaining):
            """
            Recursively check if a winning line exists in direction (dr,dc).
            remaining: how many cells including this one we still need to match.
            """
            if r < 0 or r >= self.m or c < 0 or c >= self.n:
                return False
            if self.board[r][c] != player:
                return False
            if remaining == 1:
                return True
            return check_direction(r + dr, c + dc, dr, dc, player, remaining - 1)

        for r in range(self.m):
            for c in range(self.n):
                player = self.board[r][c]
                if player:
                    if (check_direction(r, c, 1, 0, player, self.k) or
                        check_direction(r, c, 0, 1, player, self.k) or
                        check_direction(r, c, 1, 1, player, self.k) or
                        check_direction(r, c, 1, -1, player, self.k)):
                        return player
        return None


#  Experiment & testing functions

def measure_time(m, n, k, prune=False):
    """
    Measure execution time and nodes visited for a single minimax run.
    Returns tuple (time_taken, nodes_visited)
    """
    global nodes_visited
    nodes_visited = 0

    g = Game(m, n, k, perform_prune=prune)
    g.initialize_game()

    start = time.time()
    g.max()
    end = time.time()

    return end - start, nodes_visited


# Test cases 

def test_empty_board():
    """Test that minimax on an empty 3x3 board returns 0 with all moves available."""
    g = Game(3, 3, 3, True)
    g.initialize_game()
    value, moves = g.max(return_moves=True)
    assert value == 0, f"empty board value should be 0, got {value}"
    assert len(moves) == 9, f"empty board should have 9 moves, got {len(moves)}"
    assert set(moves) == g.available_cells, "recommended moves should be all cells"


def test_x_can_win_in_one():
    """Test that X can detect a winning move in one turn."""
    g = Game(3, 3, 3, True)
    g.board = [
        ["X", "X", ""],
        ["O", "O", ""],
        ["",  "",  ""],
    ]
    g.available_cells = {(r, c) for r in range(3) for c in range(3) if g.board[r][c] == ""}
    value, moves = g.max(return_moves=True)
    assert value == 1, f"X should be able to win, value should be 1, got {value}"
    assert moves == [(0, 2)], f"only winning move should be (0,2), got {moves}"


def test_min_blocks_x():
    """Test that O blocks X's immediate winning move."""
    g = Game(3, 3, 3, True)
    g.board = [
        ["X", "X", ""],
        ["",  "O", ""],
        ["",  "",  ""],
    ]
    g.available_cells = {(r, c) for r in range(3) for c in range(3) if g.board[r][c] == ""}
    _, move = g.min(return_move=True)
    assert move == (0, 2), f"Min should block at (0,2), got {move}"


def test_all_X_moves_bad():
    """Test that when all X moves are bad, all remaining cells are equally bad."""
    g = Game(3, 3, 3)
    g.board = [
        ["X", "X", "O"],
        ["X", "", ""],
        ["O", "O", ""],
    ]
    g.available_cells = {(r, c) for r in range(3) for c in range(3) if g.board[r][c] == ""}
    val, moves = g.max(return_moves=True)
    assert set(moves) == {(1, 2), (1, 1), (2, 2)}, f"All moves are equally bad"
    assert val == -1, f"Value should be -1 got {val}"


# Main execution 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run m,n,k game or experiments")
    parser.add_argument("--play", action="store_true", help="Play and test the game interactively")
    parser.add_argument("--experiments", action="store_true", help="Run experiment timings")
    parser.add_argument("--prune", action="store_true", help="Use pruning for interactive game")
    parser.add_argument("--m", type=int, default=4, help="Number of rows for interactive game")
    parser.add_argument("--n", type=int, default=4, help="Number of columns for interactive game")
    parser.add_argument("--k", type=int, default=4, help="Win condition for interactive game")
    args = parser.parse_args()

    if args.play:
        # Run all tests first
        test_empty_board()
        test_x_can_win_in_one()
        test_min_blocks_x()
        test_all_X_moves_bad()
        print("All tests passed.\n")

        # Initialize and play game
        game = Game(args.m, args.n, args.k, perform_prune=args.prune)
        game.initialize_game()
        game.play()

    elif args.experiments:
        # Run timing experiments for different board sizes and pruning
        results = []

        configs = [
            (3, 3, 3, True),   
            (3, 3, 3, False),
            (3, 4, 3, True),   
            (3, 4, 3, False),
            (4, 3, 3, True),   
            (4, 3, 3, False),
            (4, 4, 3, True),   # ONLY pruning
        ]

        for m, n, k, prune in configs:
            print(f"\nRunning {m}×{n}×{k}, prune={prune}")
            t, nodes = measure_time(m, n, k, prune)
            print(f"time taken: {t}")
            print(f"nodes visited: {nodes}")

            results.append({
                "m": m,
                "n": n,
                "k": k,
                "pruning": prune,
                "time": t,
                "nodes_visited": nodes,
            })

        df = pd.DataFrame(results)
        df.to_csv("minimax_timings.csv", index=False)
        print("\nExperiment results saved to minimax_timings.csv")
