import heapq

# Define the 8-puzzle class
class PuzzleState:
    def __init__(self, state, parent=None, move=None, g=0):
        self.state = state  # Current state of the puzzle as a tuple
        self.parent = parent  # Parent state
        self.move = move  # Move to get to this state
        self.g = g  # Cost to reach this state (steps taken)
        self.h = self.calculate_manhattan_distance()  # Heuristic cost (Manhattan distance)
        self.f = self.g + self.h  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

    # Calculate the Manhattan distance heuristic
    def calculate_manhattan_distance(self):
        distance = 0
        for i in range(9):
            if self.state[i] != 0:
                correct_position = self.state.index(self.state[i])
                goal_row, goal_col = divmod(correct_position, 3)
                current_row, current_col = divmod(i, 3)
                distance += abs(goal_row - current_row) + abs(goal_col - current_col)
        return distance

    # Get possible moves (up, down, left, right)
    def get_possible_moves(self):
        moves = []
        index = self.state.index(0)
        row, col = divmod(index, 3)

        # Define possible moves based on the position of the blank space (0)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_state = list(self.state)
                # Swap the blank space (0) with the target tile
                new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
                moves.append(tuple(new_state))
        return moves

# A* Search Algorithm
def a_star_search(initial_state, goal_state):
    open_list = []
    closed_list = set()

    # Initial state setup
    start_state = PuzzleState(initial_state)
    goal_tuple = tuple(goal_state)

    heapq.heappush(open_list, start_state)

    while open_list:
        # Get the state with the lowest f value
        current_state = heapq.heappop(open_list)

        # If we reach the goal state, return the solution path
        if current_state.state == goal_tuple:
            solution_path = []
            while current_state:
                solution_path.append(current_state.state)
                current_state = current_state.parent
            return solution_path[::-1]  # Reverse the path to get the solution

        # Add current state to closed list
        closed_list.add(tuple(current_state.state))

        # Expand the current state by generating possible moves
        for move in current_state.get_possible_moves():
            if tuple(move) in closed_list:
                continue

            # Create a new state with the current move
            new_state = PuzzleState(move, parent=current_state, g=current_state.g + 1)
            
            # Add to open list
            heapq.heappush(open_list, new_state)

    return None  # Return None if no solution found

# Function to print the puzzle state in a readable format
def print_puzzle(state):
    for i in range(0, 9, 3):
        print(f"{state[i]} {state[i+1]} {state[i+2]}")
    print()

# Example usage:
if __name__ == "__main__":
    initial_state = (2, 8, 3,
                     1, 6, 4,
                     7, 0, 5)  # Initial state of the puzzle
    goal_state = (1, 2, 3,
                  8, 0, 4,
                  7, 6, 5)  # Goal state of the puzzle

    solution = a_star_search(initial_state, goal_state)
    
    if solution:
        print("Solution found!")
        for step in solution:
            print_puzzle(step)
    else:
        print("No solution exists.")
