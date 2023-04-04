"""
settings.py
where the settings of the program should be modified.
Settings saved in 'settings.cfg' config file.
"""

import misc
import os
import json

# defaults
CONFIG_FILE = 'settings.cfg'
PROCESS_ONLINE = True
SINGLE_CLASSIFICATION = False
FPS_RATE = 10
MAX_FPS = 30
MIN_FPS = 5


def load(filename: str = CONFIG_FILE) -> None:
    """
    Load settings from a JSON configuration file.

    Parameters:
        filename: A string representing the file name of the JSON configuration file.
    """
    global PROCESS_ONLINE, FPS_RATE
    with open(filename, 'r') as file:
        settings = json.load(file)
    PROCESS_ONLINE = settings['PROCESS_ONLINE']
    FPS_RATE = settings['FPS_RATE']


def save(filename: str = CONFIG_FILE) -> None:
    """
    Save settings to a JSON configuration file.

    Parameters:
        filename: A string representing the file name of the JSON configuration file.
    """
    with open(filename, 'w') as file:
        json.dump({'PROCESS_ONLINE': PROCESS_ONLINE,
                   'FPS_RATE': FPS_RATE}, file)


def setup() -> None:
    """
    Load settings if the configuration file exists; otherwise, save default settings.
    """
    if os.path.exists('settings.cfg'):
        load()
    else:
        save()


setup()

menu_options = ['1', '2', 'M']
sub_options = ['1', '2']

menu_prompt = "1: Toggle Online/Offline Computation (currently: {})\n" \
              "3: Change FPS for live capture (currently: {})\nM: Exit Settings"


def main() -> None:
    """
    Main function for interacting with the settings menu.
    """
    while (True):
        print(menu_prompt.format(
            'online' if PROCESS_ONLINE else 'offline',
            FPS_RATE))
        print("Press the key to the corresponding action")

        reprompt = handle_key(key=misc.read_input(menu_options))
        if not reprompt:
            break

        save()


def handle_key(key: str) -> bool:
    """
    Handle the user's menu selection and call the appropriate function.

    Parameters:
        key: A string representing the user's menu selection.

    Returns:
        A boolean indicating whether to reprompt the user for input (True) or exit the menu loop (False).
    """
    if (key == -1):
        misc.print_invalid_input()
    elif (key == menu_options[0]):
        toggle_computation_mode()
    elif (key == menu_options[1]):
        toggle_classification_mode()
    elif (key == menu_options[2]):
        toggle_fps()
    else:
        misc.print_menu_return()
        return False
    return True


def toggle_computation_mode() -> None:
    """
    Toggle the computation mode between online and offline.
    """
    global PROCESS_ONLINE
    print("Do you want online or offline computation?")
    print("1: online")
    print("2: offline")

    key = misc.read_input(sub_options)

    if (key == -1):
        misc.print_invalid_input()
        return -1

    if (key == sub_options[0]):
        PROCESS_ONLINE = True
        print("Processing is online")
    else:
        PROCESS_ONLINE = False
        print("Processing is offline")


def toggle_classification_mode():
    """
    Toggle the classification mode between single and multi-object.
    """
    global SINGLE_CLASSIFICATION
    print("Do you want single or multi object classification?")
    print("1: single")
    print("2: multi")

    key = misc.read_input(sub_options)

    if (key == -1):
        misc.print_invalid_input()
        return -1

    if (key == sub_options[0]):
        SINGLE_CLASSIFICATION = True
        print("classification is singular")
    else:
        SINGLE_CLASSIFICATION = False
        print("classification is multi")


def toggle_fps() -> None:
    """
    Adjust the FPS rate for live capture.
    """
    global FPS_RATE
    print("Enter a number for the framerate for live capture")
    print("Min framerate: ", MIN_FPS)
    print("Max framerate: ", MAX_FPS)

    key = misc.read_number(low=MIN_FPS, high=MAX_FPS)

    if (key == -1):
        misc.print_invalid_input()
        return -1

    FPS_RATE = key
    print("FPS is set to", FPS_RATE)


if __name__ == "__main__":
    main()
