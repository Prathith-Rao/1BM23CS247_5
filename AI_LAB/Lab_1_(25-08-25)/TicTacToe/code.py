def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def is_moves_left(board):
    return any(cell == ' ' for row in board for cell in row)

def evaluate(board):
    for row in board:
        if row[0] == row[1] == row[2] != ' ':
            return +10 if row[0] == 'O' else -10
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return +10 if board[0][col] == 'O' else -10
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return +10 if board[0][0] == 'O' else -10
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return +10 if board[0][2] == 'O' else -10
    return 0

def minimax(board, depth, is_max):
    score = evaluate(board)
    if score == 10:
        return score - depth
    if score == -10:
        return score + depth
    if not is_moves_left(board):
        return 0
    if is_max:
        best = -1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = max(best, minimax(board, depth + 1, False))
                    board[i][j] = ' '
        return best
    else:
        best = 1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = min(best, minimax(board, depth + 1, True))
                    board[i][j] = ' '
        return best

def find_best_move(board):
    best_val = -1000
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_val = move_val
                    best_move = (i, j)
    return best_move

def main():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic Tac Toe!")
    print("You are X, AI is O")
    print_board(board)

    while True:
        while True:
            try:
                row, col = map(int, input("Enter your move (row and column 0-2, space separated): ").split())
                if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("Invalid move. Try again.")
            except:
                print("Invalid input format. Please enter two numbers between 0 and 2 separated by space.")

        print_board(board)
        if evaluate(board) == -10:
            print("You win!")
            break
        if not is_moves_left(board):
            print("It's a tie!")
            break

        print("AI is making a move...")
        ai_move = find_best_move(board)
        if ai_move != (-1, -1):
            board[ai_move[0]][ai_move[1]] = 'O'

        print_board(board)
        if evaluate(board) == 10:
            print("AI wins!")
            break
        if not is_moves_left(board):
            print("It's a tie!")
            break

if __name__ == "__main__":
    main()
