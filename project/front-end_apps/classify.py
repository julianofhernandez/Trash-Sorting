"""
classify.py
the classify menu to be used to collect the data to be classified
by the user
"""

import misc
import cv2
from camera import *
from ssd import ssd_preds

CAMERA = None

menu_options = ['1', '2', '3', 'M']

menu_prompt = menu_options[0] + ": Open Camera and capture\n" + \
    menu_options[1] + ": Upload picture\n" + \
    menu_options[2] + ": Capture real time\n" + \
    menu_options[3] + ": Exit Classify"


def main(process_online, single_classification, fps_rate):
    global CAMERA
    print("Classify trash")
    while(True):
        print(menu_prompt)
        print("Press the key to the corresponding action")

        key = misc.read_input(menu_options)

        if(key == -1):
            misc.print_invalid_input()
            continue

        if(key == menu_options[0]):
            camera_classify(process_online, single_classification)
        elif(key == menu_options[1]):
            file_classify(process_online, single_classification)
        elif(key == menu_options[2]):
            real_time_classify(process_online, single_classification, fps_rate)
        else:
            if CAMERA is not None:
                CAMERA.close()
                CAMERA = None
            misc.print_menu_return()
            return False


def camera_classify(process_online, single_classification):
    global CAMERA
    # clear screen
    print("classifying from camera")
    if CAMERA is None:
        print("Starting up camera...")
        CAMERA = CameraCapturer()
    input('Enter to capture')
    img = CAMERA.capture()
    # send img to Server or Local Model
    preds = ssd_preds(img, process_online, single_classification)
    if preds is None:
        print("Failed to classify")
    else:
        for pred in preds:
            print(pred)


def file_classify(process_online, single_classification):

    print("classifying from file")
    print("Enter image file path to be classifyed: ")
    input_file = input()
    img = cv2.imread(input_file)

    if img is None:
        print("Image could not be read")
    else:
        # send img to Server or Local Model
        preds = ssd_preds(img, process_online, single_classification)
        if preds is None:
            print("Failed to classify")
        else:
            for pred in preds:
                print(pred)


def real_time_classify(process_online, single_classification, fps_rate):
    global CAMERA
    print("classifying in real time")
    if CAMERA is not None:
        CAMERA.close()
        CAMERA = None
    print("Starting up camera...")

    # add real time camera function
    with CameraRecorder(1, fps=fps_rate) as cr:
        print("Active. To stop, press Q.")
        while True:
            img = cr.capture()
            cv2.imshow("Classifing Webcam", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            preds = ssd_preds(img, process_online, single_classification)
            if preds is None:
                print("Failed to classify")
            else:
                for pred in preds:
                    print(pred)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
