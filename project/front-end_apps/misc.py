"""
misc.py
Provides a list of helper functions to be used by other classes
Currrently provides a list of functions for reading in different inputs
"""

""" 
Reads input and returns the first character of the input if
the first character is part of a valid input. Returns -1 otherwise
for invalid inputs
"""
def read_input(valid_inputs):

    user_input = input()

    #Invalid input if input length is zero, greater than one, or not in list of valid inputs
    if(len(user_input) == 0 or len(user_input) > 1 or user_input[0].upper() not in valid_inputs):
        return -1

    return user_input[0].upper()

""" 
Reads input and tokenizes valid input. Returns -1 otherwise
for invalid inputs
"""
def read_input_tokens(valid_inputs):

    user_input = input()

    if(len(user_input) <= 0):
        return -1

    user_input = user_input.split()

    if(user_input[0].upper() not in valid_inputs):
        return -1

    user_input[0] = user_input[0].upper()

    return user_input

""" 
Reads input and returns the input if it is a numeric value 
within the range [low, high]
"""
def read_number(low, high):

    user_input = input()

    if(len(user_input) <= 0 or not user_input.isdigit()):
        return -1

    user_input = int(user_input)

    if(user_input < low or user_input > high):
        return -1

    return user_input


def print_invalid_input():
    print("ERROR: Invalid option (Enter key to the left of menu options).\n")


def print_menu_return():
    print("Returning to menu.")
