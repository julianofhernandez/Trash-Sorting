"""
annotate.py
This is the file to open up the annotation gui which will add annotations
to images for training.
"""

import misc
from camera import *
import cv2
from io import BytesIO
import requests

SECRET_KEY = 'secretkey'
HOST = 'localhost'
PORT = 5000
CAMERA = None


class Annotations:
    def __init__(self, prev_annotations=None):
        if prev_annotations is None:
            self.annotations = []
        else:
            self.annotations = prev_annotations
        self.current_annotation = [None, None]
        self.done = False
        self.active = False

    def add_annotation(self):
        self.annotations.append(self.current_annotation)

    def start_annotation(self, x, y):
        self.current_annotation[0] = (x, y)

    def temp_annotation(self, x, y):
        if self.current_annotation[0] is not None:
            self.current_annotation[1] = (x, y)
            self.active = True

    def end_annotation(self, x, y):
        self.current_annotation[1] = (x, y)
        self.annotations.append(tuple(self.current_annotation))
        self.active = False
        self.current_annotation = [None, None]

    def is_active(self):
        return self.active

    def reset_all_annotations(self):
        self.annotations = []
        self.current_annotation = [None, None]
        self.active = False

    def reset_current_annotation(self):
        self.current_annotation = [None, None]
        self.active = False

    def reset_last_annotation(self):
        del self.annotations[-1]

    def finish(self):
        self.done = True

    def is_done(self):
        return self.done


# list of valid commands the user should be able to use in this menu
menu_options = ['1', '2', 'M']

# the text prompt for this menu
menu_prompt = menu_options[0] + ": Opens GUI to capture a photo and annotate\n" + \
    menu_options[1] + ": Opens GUI and loads image from path to annotate\n" + \
    menu_options[2] + ": Exit Annotation"


def main(process_online, single_classification, fps_rate):
    global CAMERA
    while(True):
        print(menu_prompt)
        key = misc.read_input_tokens(menu_options)

        if(key == -1):
            misc.print_invalid_input()
            continue
        if(key[0] == menu_options[0]):
            img = open_from_camera()

            if img is None:
                print("image could not be read")
                continue

            annotation = handle_annotation_ui(img)

            upload_annotation(annotation, img)

        elif(key[0] == menu_options[1]):
            img = open_from_path()

            if img is None:
                print("image could not be read")
                continue

            annotation = handle_annotation_ui(img)

            upload_annotation(annotation, img)

        else:
            if CAMERA is not None:
                CAMERA.close()
                CAMERA = None
            misc.print_menu_return()
            return False


def handle_annotation_ui(img, prev_annotation=None):
    print("\n Draw: Left click drag\n Reset: Double Right Click\n Done: Ctrl + Right Click\n Exit: Esc")
    annotation = Annotations(prev_annotation)

    # define mouse callback function to draw bounding boxes
    def mouse_callback(event, x, y, flags, param):
        """
        Ref - https://www.tutorialspoint.com/opencv-python-how-to-draw-a-rectangle-using-mouse-events
        """
        nonlocal annotation
        if flags == cv2.EVENT_FLAG_CTRLKEY and event == cv2.EVENT_RBUTTONUP:
            annotation.finish()
        elif event == cv2.EVENT_LBUTTONDOWN:
            annotation.start_annotation(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            annotation.end_annotation(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            annotation.temp_annotation(x, y)
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            annotation.reset_all_annotations()

    # Create a window and bind the function to window
    cv2.namedWindow("Annotation Window")

    # Make window threaded
    cv2.startWindowThread()

    # Connect the mouse button to our callback function
    cv2.setMouseCallback("Annotation Window", mouse_callback)

    # display the window
    img_original = img.copy()
    while True:
        img = img_original.copy()
        for (x1, y1), (x2, y2) in annotation.annotations:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        if annotation.is_active():
            (x1, y1), (x2, y2) = annotation.current_annotation
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        cv2.imshow("Annotation Window", img)
        # 27 is Escape key to Discrad and exit
        if cv2.waitKey(1) == 27:
            break
        if annotation.is_done():
            break
    cv2.destroyAllWindows()
    # needed on MacOS to update window state
    cv2.waitKey(1)
    return annotation


def open_annotation(live_capture=False, path=False):
    """ Function to open up the annotation gui as referenced in 
    https://github.com/julianofhernandez/Trash-Sorting/blob/main/images/Curated%20Diagrams.pdf
    live_capture     will open it with the option of live capturing
    path        will open it with the image from the path loaded up

    Current important task - get the coordinate of the rectangle drawn with cv2

    Feature including
    - add annotation
    - reset all annotation 
    - Finish Annotation

    Use of this function is deprecated to make code more modular and testable.
    Portions of this code have been split up via functionality so this function is no
    longer useful, but is being kept here for developers
    """

    # Live capture to annotate
    if live_capture is True:
        return open_from_camera()
    elif path is True:
        return open_from_path()
    return None


def open_from_camera():
    """
    Opens up the Users camera and displays the camera on the screen. Once the user
    presses SPACEBAR, the camera capture the image and return the image.
    """
    global CAMERA
    print("Annotating from camera")
    if CAMERA is None:
        print("Starting up camera...")
        CAMERA = CameraCapturer()

    print("\nPress SPACE to capture, Press ESCAPE to exit")
    img = CAMERA.capture()

    while(img is not None):
        img = CAMERA.capture()
        cv2.imshow('Press SPACE to capture, Press ESCAPE to exit', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("\nCaptured")
            break
        elif key == 27:
            img = None
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    CAMERA.close()
    CAMERA = None

    return img


def open_from_path():
    """
    Prompts the user to open an image. Returns None if failed to open
    """

    path = input("Type in the path\n")

    return cv2.imread(path)


def upload_annotation(annotation, img):
    """
    Uploads annotations to the annotation database
    """
    if annotation.is_done():
        # Send to server
        print(annotation.annotations)
        byte_arr = BytesIO()
        byte_arr.write(cv2.imencode('.jpg', img)[1])
        byte_arr.seek(0)

        res = requests.post(f'http://{HOST}:{PORT}/create/entry', data={
            'key': SECRET_KEY,
            'annotation': annotation.annotations,
            'num_annotations': 1,
            'dataset': 'custom',
            'metadata': ''
        },
            files={'image': byte_arr})
        # Option to get entry
    else:
        print("Exited Annotation UI")
        pass


if __name__ == "__main__":
    main()
