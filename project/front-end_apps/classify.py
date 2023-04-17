"""
classify.py
the classify menu to be used to collect the data to be classified
by the user
"""

import misc
import cv2
from camera import *
from model import preds
from pprint import pprint

CAMERA = None

menu_options = ["1", "2", "3", "M"]

menu_prompt = (
    "1: Open Camera and capture\n2: Upload picture\n"
    "3: Capture real time\nM: Exit Classify"
)


def main(process_online: bool, fps_rate: int, model_offline: str) -> bool:
    """
    Main function for the classification process.

    Parameters:
        process_online: A boolean to determine if online processing is enabled.
        fps_rate: An integer representing the frames per second rate.
        model_offline: A string representing the name of the offline model.
    Returns:
        False when the user exits the menu loop.
    """
    global CAMERA
    print("Classify trash")
    while True:
        print(menu_prompt)
        print("Press the key to the corresponding action")

        key = misc.read_input(menu_options)

        if key not in menu_options:
            misc.print_invalid_input()
            continue

        if key == menu_options[0]:
            camera_classify(process_online, model_offline)
        elif key == menu_options[1]:
            file_classify(process_online, model_offline)
        elif key == menu_options[2]:
            real_time_classify(process_online, fps_rate, model_offline)
        else:
            if CAMERA is not None:
                CAMERA.close()
                CAMERA = None
            misc.print_menu_return()
            return False


def camera_classify(process_online: bool, model_offline: str) -> None:
    """
    Capture and classify an image using the camera.

    Parameters:
        process_online: A boolean to determine if online processing is enabled.
        model_offline: A string representing the name of the offline model.
    """
    global CAMERA
    # clear screen
    print("classifying from camera")
    if CAMERA is None:
        print("Starting up camera...")
        CAMERA = CameraCapturer()

    print("\nPress SPACE to capture, Press ESCAPE to exit")
    img = CAMERA.capture()

    while img is not None:
        img = CAMERA.capture()
        cv2.imshow("Press SPACE to capture, Press ESCAPE to exit", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("\nCaptured")

            # send img to Server or Local Model
            pred = preds(img, process_online, model_offline)
            if pred is None:
                print("Failed to classify")
            else:
                pprint(pred)
            break
        elif key == 27:
            img = None
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    CAMERA.close()
    CAMERA = None


def file_classify(process_online: bool, model_offline: str) -> None:
    """
    Classify an image from a file.

    Parameters:
        process_online: A boolean to determine if online processing is enabled.
        model_offline: A string representing the name of the offline model.
    """
    print("Classifying from file")
    print("Enter image file path to be classifyed: ")
    input_file = input()
    img = cv2.imread(input_file)

    if img is None:
        print("Image could not be read")
    else:
        # send img to Server or Local Model
        pred = preds(img, process_online, model_offline)
        if pred is None:
            print("Failed to classify")
        else:
            pprint(pred)


def real_time_classify(process_online: bool, fps_rate: int, model_offline: str) -> None:
    """
    Classify objects in real-time using the camera.

    Parameters:
        process_online: A boolean to determine if online processing is enabled.
        fps_rate: An integer representing the frames per second rate.
        model_offline: A string representing the name of the offline model.
    """
    global CAMERA
    print("Classifying in real time")
    if CAMERA is not None:
        CAMERA.close()
        CAMERA = None
    print("Starting up camera...")

    # add real time camera function
    with CameraRecorder(1, fps=fps_rate) as cr:
        print("Active. Press Q to stop")
        while True:
            img = cr.capture()
            cv2.imshow("Classifing. Press Q to stop", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            pred = preds(img, process_online, model_offline)
            if pred is None:
                print("Failed to classify")
            else:
                pprint(pred)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
