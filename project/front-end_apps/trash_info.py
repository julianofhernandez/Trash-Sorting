"""
trash_info.py
Used for printing out the information relating to a certain kind 
of trash type, particularly its basic info and how to dispose of it
Last modified 11/3 by Jeff de Jesus
"""

import misc

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

def main():
    while(True):
        print("Select which type of trash you want information on.")
        print(menu_prompt)

        key = misc.read_input(menu_options)

        if(key not in menu_options):
            misc.print_invalid_input()
            continue

        if(key == menu_options[0]):
            print(trash_info)

        elif(key == menu_options[1]):
            print(recycle_info)

        elif(key == menu_options[2]):
            print(compost_info)

        else:
            misc.print_menu_return()
            return False


if __name__ == "__main__":
    main()
