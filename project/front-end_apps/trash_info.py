'''
trash_info.py
Used for printing out the information relating to a certain kind 
of trash type, particularly its basic info and how to dispose of it
Last modified 11/3 by Jeff de Jesus
'''

import misc

trash_info = """Trash should be disposed of in the nearest trash bin.
Some examples of trash are, styrofoam, cooking oil, 
plastic bags, food-soiled paper, or broken ceramics.\n"""

recycle_info = """"Recycle should be placed in the nearest blue recycle bin. "
Some examples of recycle are plastic bottles, aluminum cans, "
glass bottles, cardboard, newspapers, or paper bags.\n"""

compost_info = """Compost should be disposed of in the nearest green compost bin. 
Some examples of compost are fruits, coffee grounds, tea bags, 
and egg shells. \n"""

menu_options = ['1', '2', '3', '4', 'Q']

menu_prompt = menu_options[0] + ": Trash\n" + \
              menu_options[1] + ": Recycle\n" + \
              menu_options[2] + ": Compost\n" + \
              menu_options[3] + ": Quit"


def main():
    while(True):
        print("Select which type of trash you want information on.")
        print(menu_prompt)

        key = misc.read_input(menu_options)

        if(key == -1):
            misc.print_invalid_input()
            continue

        if(key == menu_options[0]):
            print(trash_info)

        elif(key == menu_options[1]):
            print(recycle_info)

        elif(key == menu_options[2]):
            print(compost_info)

        else:
            print("Quitting\n")
            return 0


def print_trash_info(type):
    '''Will print out the info for a specific type of trash given the type
	'''
    print("Trash Type: ", type)


if __name__ == "__main__":
    main()
