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


def load(filename=CONFIG_FILE):
    global PROCESS_ONLINE, SINGLE_CLASSIFICATION, FPS_RATE
    with open(filename, 'r') as file:
        settings = json.load(file)
    PROCESS_ONLINE = settings['PROCESS_ONLINE']
    SINGLE_CLASSIFICATION = settings['SINGLE_CLASSIFICATION']
    FPS_RATE = settings['FPS_RATE']


def save(filename=CONFIG_FILE):
    with open(filename, 'w') as file:
        json.dump({'PROCESS_ONLINE': PROCESS_ONLINE,
                   'SINGLE_CLASSIFICATION': SINGLE_CLASSIFICATION,
                   'FPS_RATE': FPS_RATE}, file)


def setup():
    if os.path.exists('settings.cfg'):
        load()
    else:
        save()


setup()

menu_options = ['1', '2', '3', 'M']
sub_options = ['1', '2']

menu_prompt = menu_options[0] + ": Toggle Online/Offline Computation (currently: {})\n" + \
    menu_options[1] + ": Toggle Single/Multi object Classification (currently: {})\n" + \
    menu_options[2] + ": Change FPS for live capture (currently: {})\n" + \
    menu_options[3] + ": Exit Settings"


def main():
    while (True):
        print(menu_prompt.format(
            'online' if PROCESS_ONLINE else 'offline',
            'single' if SINGLE_CLASSIFICATION else 'multi',
            FPS_RATE))
        print("Press the key to the corresponding action")

        reprompt = handle_key(key=misc.read_input(menu_options))
        if not reprompt:
            break

        save()


def handle_key(key):
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


def toggle_computation_mode():
    """ Toggles the computation between online and offline
    """
    global PROCESS_ONLINE
    print("Do you want online or offline computation?")
    print(sub_options[0] + ": online")
    print(sub_options[1] + ": offline")

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
    """ Toggles the classification mode between single and multi
    """
    global SINGLE_CLASSIFICATION
    print("Do you want single or multi object classification?")
    print(sub_options[0] + ": single")
    print(sub_options[1] + ": multi")

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


def toggle_fps():
    """ Adjusts the FPS rate for live capture
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
