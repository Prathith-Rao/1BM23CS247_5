from collections import deque

GOAL_STATE = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))  # 0 represents empty tile

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right moves

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def swap_positions(state, pos1, pos2):
    state = [list(row) for row in state]
    state[pos1[0]][pos1[1]], state[pos2[0]][pos2[1]] = state[pos2[0]][pos2[1]], state[pos1[0]][pos1[1]]
    return tuple(tuple(row) for row in state)

def get_neighbors(state):
    zero_pos = find_zero(state)
    neighbors = []
    for move in MOVES:
        new_row, new_col = zero_pos[0] + move[0], zero_pos[1] + move[1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = swap_positions(state, zero_pos, (new_row, new_col))
            neighbors.append(new_state)
    return neighbors

def dls(state, limit, visited, path):
    if state == GOAL_STATE:
        return path + [state]
    if limit == 0:
        return None
    visited.add(state)
    for neighbor in get_neighbors(state):
        if neighbor not in visited:
            result = dls(neighbor, limit - 1, visited, path + [state])
            if result is not None:
                return result
    visited.remove(state)
    return None

def ids(start_state, max_depth=50):
    for depth in range(max_depth):
        visited = set()
        result = dls(start_state, depth, visited, [])
        if result is not None:
            return result
    return None

def print_path(path):
    for step, state in enumerate(path):
        print(f"Step {step}:")
        for row in state:
            print(row)
        print()

# Example start state
start_state = ((1, 2, 3),
               (4, 0, 6),
               (7, 5, 8))

solution_path = ids(start_state)

if solution_path:
    print(f"Solution found in {len(solution_path) - 1} moves:")
    print_path(solution_path)
else:
    print("No solution found within max depth.")
