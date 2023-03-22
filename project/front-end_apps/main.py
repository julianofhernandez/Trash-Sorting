"""
main.py
This is the main file that the end user should be running
in order to access the application
"""

import annotate
import classify
import trash_info
import about
import settings
import misc

# list of valid commands the user should be able to use in this menu
menu_options = ['1', '2', '3', '4', '5', 'Q']

# the text prompt for this menu
menu_prompt = "1: Annotate\n2: Classify\n3: Trash info\n4: About\n5: Settings\nQ: Quit"

def main() -> int:
    """
    This is the main function. This version will only
    read in the first character of each input line and treat
    those inputs as valid by trimming the rest of the input
    """
    print("Main Menu")

    while True:
        print(menu_prompt)
        print("Press the key to the corresponding action.")

        # collect input from user, if input is invalid -1 is returned
        key = misc.read_input(menu_options)

        if key not in menu_options:
            misc.print_invalid_input()
            continue

        if key == menu_options[0]:
            # open annotate
            annotate.main(settings.PROCESS_ONLINE, settings.SINGLE_CLASSIFICATION, settings.FPS_RATE)
        elif key == menu_options[1]:
            # open classify
            classify.main(settings.PROCESS_ONLINE, settings.SINGLE_CLASSIFICATION, settings.FPS_RATE)
        elif key == menu_options[2]:
            # open trash info
            trash_info.main()
        elif key == menu_options[3]:
            # open about
            about.main()
        elif key == menu_options[4]:
            # open settings
            settings.main()
        else:
            # quit program
            print("Goodbye")
            return 0


if __name__ == "__main__":
    main()