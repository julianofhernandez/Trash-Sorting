"""
trash_info.py
Used for printing out the information relating to a certain kind 
of trash type, particularly its basic info and how to dispose of it
"""

import misc
from typing import Union

trash_info = """Trash should be disposed of in the nearest trash bin.
Some examples of trash are, styrofoam, cooking oil, 
plastic bags, food-soiled paper, or broken ceramics.\n"""

recycle_info = """Recycle should be placed in the nearest blue recycle bin.
Some examples of recycle are plastic bottles, aluminum cans,
glass bottles, cardboard, newspapers, or paper bags.\n"""

compost_info = """Compost should be disposed of in the nearest green compost bin. 
Some examples of compost are fruits, coffee grounds, tea bags, 
and egg shells. \n"""

menu_options = ['1', '2', '3', 'M']

menu_prompt = "1: Trash\n2: Recycle\n3: Compost\nM: Exit Trash Info"


def main() -> Union[bool, None]:
    """
    Main function to run the trash info application.
    Prompts the user to select a trash type, and then displays relevant information.
    
    Returns:
        False if the user selects 'Exit Trash Info', None otherwise.
    """
    while True:
        print("Select which type of trash you want information on.")
        print(menu_prompt)

        # Read user input
        key = misc.read_input(menu_options)

        # If input is invalid, print an error message and prompt again
        if(key not in menu_options):
            misc.print_invalid_input()
            continue

        # Display trash info based on user input
        if(key == menu_options[0]):
            print(trash_info)
        elif(key == menu_options[1]):
            print(recycle_info)
        elif(key == menu_options[2]):
            print(compost_info)

        # Exit the application if the user selects 'Exit Trash Info'
        else:
            misc.print_menu_return()
            return False


if __name__ == "__main__":
    main()
