"""
Author: Travis Hammond
Version: 4_28_2022
"""

from threading import Thread, Event
from collections import deque

import numpy as np
import cv2
import os


class CameraCapturer():
    """Class for capturing frames from a camera"""

    def __init__(self, camera_device=0, fps=30, frame_width=None, frame_height=None):
        """Initialize the CameraCapturer object.

        Args:
            camera_device (int, optional): Index of the camera device to use. Defaults to 0.
            fps (int, optional): Frames per second to capture. Defaults to 30.
            frame_width (int, optional): Width of the captured frames. Defaults to None.
            frame_height (int, optional): Height of the captured frames. Defaults to None.
        """
        super().__init__()
        self.camera_device = camera_device
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        # CAP_MSMF is for Microsoft Media Foundation API so disable it for other systems
        # 'nt' means OS is Windows
        if os.name == 'nt':
            self.camera = cv2.VideoCapture(self.camera_device, cv2.CAP_MSMF)
        else:
            self.camera = cv2.VideoCapture(self.camera_device)

        if self.frame_width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)

    def close(self):
        """Release the camera resource."""
        self.camera.release()

    def capture(self):
        """Capture a single frame from the camera.

        Returns:
            numpy.ndarray: The captured frame as a numpy array.
        """
        grabbed, frame = self.camera.read()
        if not grabbed:
            raise Exception(f'Could not read frame from '
                            f'Camera {self.camera_device}')
        return frame


class CameraRecorder:
    """Class for recording frames from a camera."""

    def __init__(self, max_buffer_seconds, camera_device=0, fps=30, frame_width=None, frame_height=None):
        """Initialize the CameraRecorder object.

        Args:
            max_buffer_seconds (int): Maximum number of seconds to keep recorded frames.
            camera_device (int, optional): Index of the camera device to use. Defaults to 0.
            fps (int, optional): Frames per second to capture. Defaults to 30.
            frame_width (int, optional): Width of the captured frames. Defaults to None.
            frame_height (int, optional): Height of the captured frames. Defaults to None.
        """
        self.fps = fps
        self.camera_device = camera_device
        self.max_buffer_seconds = max_buffer_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.frames = deque(maxlen=int(self.max_buffer_seconds * self.fps))
        self.stop_event = Event()

    def clear(self):
        """Clear all the recorded frames."""
        self.frames.clear()

    def __enter__(self):
        """Enter method to be used with the "with" statement."""
        self.start_recording()
        return self

    def __exit__(self):
        """Exit method to be used with the "with" statement."""
        self.stop_recording()

    def start_recording(self):
        """Start recording frames."""
        self.stop_event.clear()

        # CAP_MSMF is for Microsoft Media Foundation API so disable it for other systems
        # 'nt' means OS is Windows
        if os.name == 'nt':
            self.camera = cv2.VideoCapture(self.camera_device, cv2.CAP_MSMF)
        else:
            self.camera = cv2.VideoCapture(self.camera_device)

        if self.frame_width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)

        self.thread = Thread(target=self._record)
        self.thread.start()

    def stop_recording(self):
        """
        Stops recording video from the camera.
        """
        self.stop_event.set()
        self.thread.join()
        self.camera.release()

    def _record(self):
        """
        Records video from the camera and adds frames to the buffer.
        """
        while not self.stop_event.is_set():
            grabbed, frame = self.camera.read()
            if grabbed:
                self.frames.append(frame)

    def get_last_frames(self, seconds=None):
        """
        Returns the frames from the last `seconds` seconds.

        Args:
            seconds (int, optional): The number of seconds to return frames for. If None, returns all frames.

        Returns:
            list: A list of frames.
        """
        if seconds is None:
            return list(self.frames)
        else:
            return list(self.frames)[-int(self.fps * seconds):]

    def capture(self):
        """
        Captures the most recent frame from the camera.

        Returns:
            numpy.ndarray: A frame from the camera.
        """
        while True:
            if len(self.frames) > 0:
                return self.frames[-1]
