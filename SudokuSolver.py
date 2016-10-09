#This file solves a sudoku given as image using neural networks. Neural networks are pretrained using digit classifier.
# Author - Naveen
from PIL import Image
from PIL import ImageOps
import PIL
import numpy as np
from sklearn.preprocessing import normalize
import pickle
import copy

pil_im = Image.open('sudoku.png')

(width, height) = pil_im.size

x_inc = int(width/9)
y_inc = int(height/9)
x_index = 0
y_index = 0

in_num = []
sudoku = []
# Initializing blank sudoku array
for r_index in range(9):
    sudoku.append([])
    for c_index in range(9):
        sudoku[r_index].append('')
#print(pil_im.size)

# Scanning the image image cell by cell on each row and column and extracting numbers from image using neural nets.
for yaxis in range(9):
    for xaxis in range(9):
        dim = (x_index, y_index, x_index + x_inc, y_index + y_inc)
        part = pil_im.crop(dim)
        re_size = part.resize((28, 28), PIL.Image.ANTIALIAS)
        arr = np.array(re_size)
        arr[:, 0] = 0
        arr[:, 27] = 0
        arr[0, :] = 0
        arr[27, :] = 0
        n_ar = normalize(arr)
        n_ar = n_ar.flatten()

        firstnet = pickle.load(open("first.pkl", "rb"))
        predicted = firstnet.predict(n_ar) # predicting a number using pre trained network
        #print(np.argmax(predicted))
        in_num.append(np.argmax(predicted))
        sudoku[yaxis][xaxis] = np.argmax(predicted) # adding predicted number to sudoku for solving
        x_index += x_inc
    x_index = 0
    y_index += y_inc


# provides numbers that are feasible foe a given cell
def get_possible_numbers_for_cell(row, col, puzzle):
    possible_numbers = []
    #print(puzzle[row][col])
    if puzzle[row][col] == 0:
        box_row_start = row - (row % 3)
        box_col_start = col - (col % 3)
        num_in_row = []
        num_in_col = []
        num_in_box = []
        for index in range(9):
            # scanning row
            num_in_row.append(puzzle[row][index])
            #scanning column
            num_in_col.append(puzzle[index][col])
        for r_index in range(3):
            for c_index in range(3):
                num_in_box.append(puzzle[box_row_start + r_index][box_col_start + c_index])

        for num in range(1,10):
            if (num not in num_in_row) and (num not in num_in_col) and (num not in num_in_box):
                possible_numbers.append(num)
    return possible_numbers

# validates input puzzle
def is_valid(puzzle):

    for index in range(9):
        row = puzzle[index]
        row_list = []
        for item in row:
            if item != 0:
                row_list.append(item)
        if len(row_list) != len(set(row_list)):  # scanning row for duplicates
            return False

    for c_index in range(9):
        col_list = []
        for r_index in range(9):
            if puzzle[r_index][c_index] != 0:
                col_list.append(puzzle[r_index][c_index])

        if len(col_list) != len(set(col_list)):  # scanning column for duplicates.
            return False

    for r_index in range(0, 9, 3):
        for c_index in range(0, 9, 3):
            box_list = []
            for box_r_index in range(3):
                for box_c_index in range(3):
                    item = puzzle[r_index + box_r_index][c_index + box_c_index]
                    if item != 0:
                        box_list.append(item)

            if len(box_list) != len(set(box_list)):  # scanning box for duplicates.
                return False

    return True

# checks if a given puzzle is already solved
def is_solved(puzzle):
    solved = True

    for r_index in range(9):
        for c_index in range(9):
            if puzzle[r_index][c_index] == 0: # scanning for unsolved cells
                solved = False
                break

    if solved and not is_valid(puzzle): # if solved then check valid
        solved = False

    return solved

# fills possible options for all cells
def get_possible_options(puzzle):
    possiblity_arr = []
    for r_index in range(9):
        possiblity_arr.append([])
        for c_index in range(9):
            possiblity_arr[r_index].append([])

    for r_index in range(9):
        for c_index in range(9):
            possiblity_arr[r_index][c_index] = get_possible_numbers_for_cell(r_index, c_index, puzzle)

    return possiblity_arr

# validates each possible option and returns if feasible
def feasible_option(updated_puzzle, options):
    feasible = True
    for r_index in range(9):
        for c_index in range(9):
            if updated_puzzle[r_index][c_index] == 0 and len(options[r_index][c_index]) == 0:
                feasible = False
                return feasible
    return feasible

# solves a sudoku for given 9 * 9 array
def solve(puzzle):
    if is_solved(puzzle):
        return puzzle

    max_possible_length = 9

    possibility_arr = get_possible_options(puzzle)

    no_guess = False
    # updating single options first
    for r_index in range(9):
        for c_index in range(9):
            if len(possibility_arr[r_index][c_index]) == 1:
                puzzle[r_index][c_index] = possibility_arr[r_index][c_index][0]
                no_guess = True

    if no_guess: # if some values are known for sure solve without guess
        puzzle = solve(puzzle)

    else:
        curr_possibility_index = 2

        while curr_possibility_index <= max_possible_length:
            curr_option_index = 1
            while curr_option_index <= curr_possibility_index: # to check options one by one
                for r_index in range(9):
                    for c_index in range(9):
                        if len(possibility_arr[r_index][c_index]) == curr_possibility_index:
                            copy = np.array(puzzle)
                            copy[r_index][c_index] = possibility_arr[r_index][c_index][curr_option_index]
                            test_options = get_possible_options(copy)
                            if feasible_option(copy, test_options):
                                solved_sudo = solve(copy)
                                if is_solved(solved_sudo):
                                    return solved_sudo
                curr_option_index += 1
            curr_possibility_index += 1
    return puzzle



print("Input:")
print(np.array(sudoku))

#solving sudoku
arr_for_solving = np.array(sudoku)

valid = is_valid(arr_for_solving)
print('Is valid puzzle:' + str(valid))
if valid:
    solved = solve(arr_for_solving)
    print("Output:")
    print(solved)
else:
    print('Invalid puzzzzzzzllllleeeee')
