"""
misc.py
Provides a list of helper functions to be used by other classes
Currrently provides a list of functions for reading in different inputs
"""


from typing import List, Tuple, Union


def read_input(valid_inputs: List[str]) -> Union[str, int]:
    """
    Reads input from the user and returns the first character of the input if it's part of a valid input, otherwise returns -1.

    Parameters:
        valid_inputs (List[str]): A list of valid inputs.

    Returns:
        Union[str, int]: A single character string if it's a valid input, otherwise -1.
    """
    user_input = input().strip().upper()

    # Invalid input if input length is zero, greater than one, or not in list of valid inputs
    if not user_input or user_input[0] not in valid_inputs:
        return -1

    return user_input[0]


def read_input_tokens(valid_inputs: List[str]) -> Union[List[str], int]:
    """
    Reads input from the user and tokenizes it if the first word is a valid input, otherwise returns -1.

    Parameters:
        valid_inputs (List[str]): A list of valid inputs.

    Returns:
        Union[List[str], int]: A list of tokens if the first token is a valid input, otherwise -1.
    """
    user_input = input().strip().upper()

    if not user_input:
        return -1

    tokens = user_input.split()

    if tokens[0] not in valid_inputs:
        return -1

    return tokens


def read_number(low: int, high: int) -> Union[int, float]:
    """
    Reads input from the user and returns it as an integer if it's a valid numeric value within the range [low, high], otherwise returns -1.

    Parameters:
        low (int): The minimum valid value.
        high (int): The maximum valid value.

    Returns:
        Union[int, float]: Returns an integer if it's a valid numeric value within the range [low, high], otherwise -1.
    """
    user_input = input().strip()

    if not user_input or not user_input.isdigit():
        return -1

    user_input = int(user_input)

    if user_input < low or user_input > high:
        return -1

    return user_input


def print_invalid_input() -> None:
    """
    Prints an error message for an invalid input.
    """
    print("ERROR: Invalid option (Enter key to the left of menu options).\n")


def print_menu_return() -> None:
    """
    Prints a message indicating a return to the menu.
    """
    print("Returning to menu.")
