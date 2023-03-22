"""
annotate.py
This is the file to open up the annotation gui which will add annotations
to images for training.
"""

import misc
from camera import *
import cv2
import requests
from typing import List, Tuple, Union, Optional

SECRET_KEY = 'secretkey'
HOST = 'localhost'
PORT = 5000
CAMERA = None


class Annotations:
    def __init__(self, prev_annotations: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None) -> None:
        """
        Parameters:
            prev_annotations: A list of previous annotations as tuples containing two tuples (x, y) coordinates.
                              Default is None, which initializes an empty list of annotations.
        """
        if prev_annotations is None:
            self.annotations = []
        else:
            self.annotations = prev_annotations
        self.current_annotation = [None, None]
        self.done = False
        self.active = False

    def add_annotation(self) -> None:
        """
        Appends the current annotation to the list of annotations.
        """
        self.annotations.append(self.current_annotation)

    def start_annotation(self, x: int, y: int) -> None:
        """
        Parameters:
            x: The x-coordinate of the starting point.
            y: The y-coordinate of the starting point.
        """
        self.current_annotation[0] = (x, y)

    def temp_annotation(self, x: int, y: int) -> None:
        """
        Sets the temporary end point of the current annotation, if the starting point is not None.
        
        Parameters:
            x: The x-coordinate of the temporary end point.
            y: The y-coordinate of the temporary end point.
        """
        if self.current_annotation[0] is not None:
            self.current_annotation[1] = (x, y)
            self.active = True

    def end_annotation(self, x: int, y: int) -> None:
        """
        Sets the end point of the current annotation and appends it to the list of annotations.
        
        Parameters:
            x: The x-coordinate of the end point.
            y: The y-coordinate of the end point.
        """
        self.current_annotation[1] = (x, y)
        self.annotations.append(tuple(self.current_annotation))
        self.active = False
        self.current_annotation = [None, None]

    def is_active(self) -> bool:
        """
        Checks if the current annotation is active.

        Returns:
            True if active, otherwise False.
        """
        return self.active

    def reset_all_annotations(self) -> None:
        """
        Resets all annotations by clearing the annotations list and setting the current annotation to [None, None].
        """
        self.annotations = []
        self.current_annotation = [None, None]
        self.active = False

    def reset_current_annotation(self) -> None:
        """
        Resets the current annotation by setting its starting and ending points to None and setting active to False.
        """
        self.current_annotation = [None, None]
        self.active = False

    def reset_last_annotation(self) -> None:
        """
        Removes the last annotation from the list of annotations.
        """
        del self.annotations[-1]

    def finish(self) -> None:
        """
        Sets the done attribute to True, indicating that the annotation process is complete.
        """
        self.done = True

    def is_done(self) -> bool:
        """
        Checks if the annotation process is done.

        Returns:
            True if done, otherwise False.
        """
        return self.done


# list of valid commands the user should be able to use in this menu
menu_options = ['1', '2', 'M']

# the text prompt for this menu
menu_prompt = "1: Opens GUI to capture a photo and annotate\n" \
              "2: Opens GUI and loads image from path to annotate\nM: Exit Annotation"

def main(process_online: bool, single_classification: bool, fps_rate: int) -> bool:
    """
    The main function that runs the menu loop for the annotation tool.
    
    Parameters:
        process_online: A boolean to determine if online processing is enabled.
        single_classification: A boolean to determine if single classification mode is enabled.
        fps_rate: An integer representing the frames per second rate.
    Returns:
        False when the user exits the menu loop.
    """
    global CAMERA
    while(True):
        print(menu_prompt)
        key = misc.read_input_tokens(menu_options)

        if(key == -1):
            misc.print_invalid_input()
            continue
        if(key[0] == menu_options[0]):
            image = open_from_camera()

            if image is None:
                print("Image could not be read")
                continue

            annotation = handle_annotation_ui(image)
            upload_annotation(annotation, image)
        elif(key[0] == menu_options[1]):
            image = open_from_path()

            if image is None:
                print("image could not be read")
                continue

            annotation = handle_annotation_ui(image)
            upload_annotation(annotation, image)
        else:
            if CAMERA is not None:
                CAMERA.close()
                CAMERA = None
            misc.print_menu_return()
            return False


def handle_annotation_ui(image: np.ndarray, prev_annotation: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None) -> Annotations:
    """
    Handles the user interface for annotation, allowing the user to draw bounding boxes on the image.

    Parameters:
        image: A numpy.ndarray representing the image to be annotated.
        prev_annotation: A list of previous annotations as tuples containing two tuples (x, y) coordinates.
                         Default is None.
    Returns:
        An Annotations object containing the annotations made by the user.
    """
    print("\n Draw: Left click drag\n Reset: Double Right Click\n Done: Ctrl + Right Click\n Exit: Esc")
    annotation = Annotations(prev_annotation)

    # define mouse callback function to draw bounding boxes
    def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
        """
        The callback function for mouse events in the annotation user interface.
        Ref - https://www.tutorialspoint.com/opencv-python-how-to-draw-a-rectangle-using-mouse-events
        
        Parameters:
            event: The mouse event.
            x: The x-coordinate of the event.
            y: The y-coordinate of the event.
            flags: Additional flags related to the event.
            param: Additional parameters related to the event.
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
    image_original = image.copy()
    while True:
        image = image_original.copy()
        for (x1, y1), (x2, y2) in annotation.annotations:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        if annotation.is_active():
            (x1, y1), (x2, y2) = annotation.current_annotation
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        cv2.imshow("Annotation Window", image)
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


def open_from_camera() -> Union[None, np.ndarray]:
    """
    Opens the user's camera and displays the camera feed. Captures the image when the user
    presses the SPACEBAR and returns the image.
    
    Returns:
        A numpy.ndarray representing the captured image, or None if no image was captured.
    """
    global CAMERA
    print("Annotating from camera")
    if CAMERA is None:
        print("Starting up camera...")
        CAMERA = CameraCapturer()

    print("\nPress SPACE to capture, Press ESCAPE to exit")
    image = CAMERA.capture()

    while(image is not None):
        image = CAMERA.capture()
        cv2.imshow('Press SPACE to capture, Press ESCAPE to exit', image)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("\nCaptured")
            break
        elif key == 27:
            image = None
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    CAMERA.close()
    CAMERA = None

    return image


def open_from_path():
    """
    Prompts the user to open an image. Returns None if failed to open
    """

    path = input("Type in the path\n")

    return cv2.imread(path)


def upload_annotation(annotation: Annotations, image: np.ndarray) -> None:
    """
    Uploads the annotations to the annotation database.

    Parameters:
        annotation: An Annotations object containing the annotations to be uploaded.
        image: A numpy.ndarray representing the image with annotations.
    """
    if annotation.is_done():
        # Send to server
        print(annotation.annotations)

        res = requests.post(f'http://{HOST}:{PORT}/create/entry', data={
            'key': SECRET_KEY,
            'annotation': annotation.annotations,
            'num_annotations': 1,
            'dataset': 'custom',
            'metadata': ''
        },
            files={'image': image})
        # Option to get entry
    else:
        print("Exited Annotation UI")
        pass


if __name__ == "__main__":
    main()
