import mcts_cpp

# This is a helper function usually provided by your SFEN parser or nshogi bindings
# Assuming you have a way to create a State from SFEN:
state = mcts_cpp.GameState.from_sfen("4k4/4g4/9/9/9/9/9/9/9 b - 1")

moves = state.legal_moves()
print(f"Number of legal moves: {len(moves)}")
print(f"Is terminal: {state.is_terminal()}")

if len(moves) == 0 and state.is_terminal():
    print("Test Passed: Terminal state correctly identified.")
