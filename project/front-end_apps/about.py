"""
about.py
This gives the about informational section of the program
"""

import misc

about_info = """This program is intended to classify trash types in order to determine which 
trash bin it belongs in. Classifications are done through the use of a computer's webcam, and 
our program has been tested to be at least 75% accurate with it's results."""

menu_info = """1. Annotate: This is used for adding annotations to images for training.
2. Classify: This is used to collect data that will be classified.
3. About: (Current selection) This is used to give users information on the 
application, explaining what each menu option does, and list developers.
4. Settings: This is used to change the settings for the application.
5. Quit: This is used to exit out of the application."""

credit_info = """Developers: Jeffrey de Jesus, Bryan Burch, Christopher Allen, 
Daniel Smagly, Julian Hernandez, Kenta Miyahara, Santiago Bermudez, 
Travis Hammond

Project Owner: Dr. Clark Fitzgerald
"""

# Acceptable menu options.
menu_options = ["1", "2", "3", "M"]

# Text prompt for each menu option.
menu_prompt = "1: About Information\n2: Menu Information\n" "3: Credits\nM: Exit About"


def main():
    """
    This function is the main entry point to the program. It provides information to the user
    about the application and allows the user to select what they want information on.
    """
    print("About: ")
    while True:
        print("Select what you want information on.")
        print(menu_prompt)

        reprompt = handle_key(key=misc.read_input(menu_options))
        if not reprompt:
            break


def handle_key(key: str) -> bool:
    """
    This function handles the selected menu option and provides the appropriate information to the user.

    Parameters:
        key (str): The selected menu option.

    Returns:
        bool: A flag indicating whether the user should be prompted for more input.
    """
    if key not in menu_options:
        misc.print_invalid_input()

    elif key == menu_options[0]:
        print(about_info + "\n")

    elif key == menu_options[1]:
        print(menu_info + "\n")

    elif key == menu_options[2]:
        print(credit_info)

    else:
        misc.print_menu_return()
        return False
    return True


if __name__ == "__main__":
    main()
