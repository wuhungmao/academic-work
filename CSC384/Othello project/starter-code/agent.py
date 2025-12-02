"""
An AI player for Othello. 
"""
import heapq
import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

Cache = {}
def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT
    val1 = get_score(board)[0]
    val2 = get_score(board)[1]
    if color == 1:
        return val1 - val2
    else:
        return val2 - val1

# Better heuristic value of board
def compute_heuristic(board, color): #not implemented, optional
    height = len(board)
    width = len(board[0])
    extra_score = 0
    for i in range(0, height):
        for j in [0, width]:
            if board[i][j] == color:
                extra_score+=1
    for i in [0, height]:
        for j in range(0, width):
            if board[i][j] == color:
                extra_score+=1

    return compute_utility(board, color) + extra_score

############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    #IMPLEMENT (and replace the line below)
    if caching == 1:
        if board in Cache:
            return (None, Cache[board])
    if color == 1:
        successor = get_possible_moves(board, 2)
    else:
        successor = get_possible_moves(board, 1)
    if len(successor) == 0 or limit == 0:
        return (None, compute_utility(board, color))
    minimized_max = float('inf')
    final_move = (0,0)
    for move in successor:
        new_board = board[:]
        if color == 1:
            board_after_move = play_move(new_board, 2, move[0], move[1])
        else:
            board_after_move = play_move(new_board, 1, move[0], move[1])
        intermidiate = minimax_max_node(board_after_move, color, limit-1, caching)
        if caching == 1:
            Cache[board_after_move] = intermidiate[1]
        if minimized_max > intermidiate[1]:
            minimized_max = intermidiate[1]
            final_move = move
    return (final_move, minimized_max)

def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    #IMPLEMENT (and replace the line below)
    if caching == 1:
        if board in Cache:
            return (None, Cache[board])
    successor = get_possible_moves(board, color)
    if len(successor) == 0 or limit == 0:
        return (None, compute_utility(board, color))
    maximized_min = -float('inf')
    final_move = (0,0)
    for move in successor:
        new_board = board[:]
        board_after_move = play_move(new_board, color, move[0], move[1])
        intermidiate = minimax_min_node(board_after_move, color, limit-1, caching)
        if caching == 1:
            Cache[board_after_move] = intermidiate[1]
        if maximized_min < intermidiate[1]:
            maximized_min = intermidiate[1]
            final_move = move
    return (final_move, maximized_min)

def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    #IMPLEMENT (and replace the line below)
    #for debugging purpose, 

    final_move = minimax_max_node(board, color, limit, caching)[0]
    return final_move
    

############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT (and replace the line below)
    if caching == 1:
        if board in Cache:
            return (None, Cache[board])
    if color == 1:
        successor = get_possible_moves(board, 2)
    else:
        successor = get_possible_moves(board, 1)
    if len(successor) == 0 or limit == 0:
        return (None, compute_utility(board, color))
    final_move = None
    new_board = board[:]
    max_heap = []
    heapq.heapify(max_heap)
    if ordering == 1:
        for move in successor:
            if color == 1:
                board_after_move = play_move(new_board, 2, move[0], move[1])
                heapq.heappush(max_heap, (-compute_utility(board_after_move, color), board_after_move))
            else:
                board_after_move = play_move(new_board, 1, move[0], move[1])
                heapq.heappush(max_heap, (-compute_utility(board_after_move, color), board_after_move))
        for i in range(0, len(max_heap)):
            max_util_board = heapq.heappop(max_heap)[1]
            intermediate = alphabeta_max_node(max_util_board, color, alpha, beta, limit-1, caching, ordering)
            if max_util_board not in Cache:
                Cache[max_util_board] = intermediate[1]
            if beta > intermediate[1]:
                final_move = move
                beta = intermediate[1]
                if beta <= alpha:
                    break
    else:
        for move in successor:
            if color == 1:
                board_after_move = play_move(new_board, 2, move[0], move[1])
            else:
                board_after_move = play_move(new_board, 1, move[0], move[1])
            intermediate = alphabeta_max_node(board_after_move, color, alpha, beta, limit-1, caching, ordering)
            if board_after_move not in Cache:
                Cache[board_after_move] = intermediate[1]
            if beta > intermediate[1]:
                final_move = move
                beta = intermediate[1]
                if beta <= alpha:
                    break
    return (final_move, beta) #change this!

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT (and replace the line below)
    if caching == 1:
        if board in Cache:
            return (None, Cache[board])
    successor = get_possible_moves(board, color)
    if len(successor) == 0 or limit == 0:
        return (None, compute_utility(board, color))
    final_move = None
    #what does it mean if all intermidiate are all smaller than alpha? what should final_move be
    new_board = board[:]
    max_heap = []
    heapq.heapify(max_heap)
    if ordering == 1:
        for move in successor:
            board_after_move = play_move(new_board, color, move[0], move[1])
            heapq.heappush(max_heap, (-compute_utility(board_after_move, color), board_after_move))
        for i in range(0, len(max_heap)):
            max_util_board = heapq.heappop(max_heap)[1]
            intermediate = alphabeta_min_node(max_util_board, color, alpha, beta, limit-1, caching, ordering)
            if max_util_board not in Cache:
                Cache[max_util_board] = intermediate[1]
            if alpha < intermediate[1]:
                final_move = move
                alpha = intermediate[1]
                if beta <= alpha:
                    break
    else:
        for move in successor:
            board_after_move = play_move(new_board, color, move[0], move[1])
            intermediate = alphabeta_min_node(board_after_move, color, alpha, beta, limit-1, caching, ordering)
            if board_after_move not in Cache:
                Cache[board_after_move] = intermediate[1]
            if alpha < intermediate[1]:
                final_move = move
                alpha = intermediate[1]
                if beta <= alpha:
                    break
    return (final_move, alpha)

def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    #IMPLEMENT (and replace the line below)
    alpha = -float('inf')
    beta = float('inf')
    final_move = alphabeta_max_node(board, color, alpha, beta, limit, caching, ordering)[0]
    return final_move

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
