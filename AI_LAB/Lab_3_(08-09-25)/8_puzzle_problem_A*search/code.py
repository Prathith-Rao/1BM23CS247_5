import heapq

# Define the Puzzle State class
class PuzzleState:
    goal_positions = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        8: (1, 0), 0: (1, 1), 4: (1, 2),
        7: (2, 0), 6: (2, 1), 5: (2, 2)
    }

    def __init__(self, state, parent=None, move=None, g=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.g = g  # Cost so far
        self.h = self.calculate_manhattan_distance()
        self.f = self.g + self.h

    def __lt__(self, other):
        return self.f < other.f

    def calculate_manhattan_distance(self):
        distance = 0
        for i, tile in enumerate(self.state):
            if tile != 0:
                goal_row, goal_col = PuzzleState.goal_positions[tile]
                current_row, current_col = divmod(i, 3)
                distance += abs(goal_row - current_row) + abs(goal_col - current_col)
        return distance

    def get_possible_moves(self):
        moves = []
        index = self.state.index(0)
        row, col = divmod(index, 3)

        directions = {
            "Up": (-1, 0),
            "Down": (1, 0),
            "Left": (0, -1),
            "Right": (0, 1)
        }

        for move, (dr, dc) in directions.items():
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_state = list(self.state)
                # Swap blank with target tile
                new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
                moves.append((tuple(new_state), move))
        return moves


def reconstruct_path(state):
    path = []
    moves = []
    while state:
        path.append(state.state)
        if state.move:
            moves.append(state.move)
        state = state.parent
    return path[::-1], moves[::-1]


def a_star_search(initial_state, goal_state):
    open_list = []
    closed_list = set()

    start_state = PuzzleState(initial_state)
    goal_tuple = tuple(goal_state)

    heapq.heappush(open_list, start_state)

    while open_list:
        current_state = heapq.heappop(open_list)

        if current_state.state == goal_tuple:
            return reconstruct_path(current_state)

        closed_list.add(current_state.state)

        for move_state, move in current_state.get_possible_moves():
            if move_state in closed_list:
                continue

            new_state = PuzzleState(move_state, parent=current_state, move=move, g=current_state.g + 1)

            # Avoid adding worse duplicates
            if any(open_state.state == new_state.state and open_state.g <= new_state.g for open_state in open_list):
                continue

            heapq.heappush(open_list, new_state)

    return None, None


# Print puzzle nicely
def print_puzzle(state):
    for i in range(0, 9, 3):
        print(" ".join(str(x) if x != 0 else " " for x in state[i:i+3]))
    print()


# Example usage
if __name__ == "__main__":
    initial_state = (2, 8, 3,
                     1, 6, 4,
                     7, 0, 5)

    goal_state = (1, 2, 3,
                  8, 0, 4,
                  7, 6, 5)

    solution, moves = a_star_search(initial_state, goal_state)

    if solution:
        print("Solution found!")
        print(f"Number of moves: {len(moves)}")
        print("Moves:", " -> ".join(moves))
        print("\nStep-by-step:")
        for step in solution:
            print_puzzle(step)
    else:
        print("No solution exists.")
