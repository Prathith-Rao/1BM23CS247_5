from collections import deque

# Define the goal state
GOAL_STATE = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))  # 0 represents the empty tile

# Moves: up, down, left, right (row, col)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def find_zero(state):
    """Find the position of the zero tile in the state."""
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def swap_positions(state, pos1, pos2):
    """Swap two positions in the puzzle state and return a new state."""
    state = [list(row) for row in state]
    state[pos1[0]][pos1[1]], state[pos2[0]][pos2[1]] = state[pos2[0]][pos2[1]], state[pos1[0]][pos1[1]]
    return tuple(tuple(row) for row in state)

def get_neighbors(state):
    """Generate all valid neighbors of the current state."""
    zero_pos = find_zero(state)
    neighbors = []
    for move in MOVES:
        new_row, new_col = zero_pos[0] + move[0], zero_pos[1] + move[1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = swap_positions(state, zero_pos, (new_row, new_col))
            neighbors.append(new_state)
    return neighbors

def dfs(start_state):
    """Perform DFS to find the goal state."""
    stack = [(start_state, [])]  # Stack holds tuples of (state, path_to_state)
    visited = set()

    while stack:
        state, path = stack.pop()

        if state == GOAL_STATE:
            return path + [state]

        if state in visited:
            continue
        visited.add(state)

        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                stack.append((neighbor, path + [state]))

    return None

def print_path(path):
    for step, state in enumerate(path):
        print(f"Step {step}:")
        for row in state:
            print(row)
        print()

# Example starting state
start_state = ((1, 2, 3),
               (4, 0, 6),
               (7, 5, 8))

solution_path = dfs(start_state)

if solution_path:
    print(f"Solution found in {len(solution_path) - 1} moves:")
    print_path(solution_path)
else:
    print("No solution found.")
