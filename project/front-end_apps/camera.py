"""
Author: Travis Hammond
Version: 4_28_2022
"""

from threading import Thread, Event
from collections import deque
from time import sleep, time

import numpy as np
import cv2
import os


class CameraCapturer():
    def __init__(self, camera_device=0, fps=30, frame_width=None, frame_height=None):
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
        self.camera.release()

    def capture(self):
        grabbed, frame = self.camera.read()
        if not grabbed:
            raise Exception(f'Could not read frame from '
                            f'Camera {self.camera_device}')
        return frame


class CameraRecorder:
    def __init__(self, max_buffer_seconds, camera_device=0, fps=30, frame_width=None, frame_height=None):
        self.fps = fps
        self.camera_device = camera_device
        self.max_buffer_seconds = max_buffer_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.frames = deque(maxlen=int(self.max_buffer_seconds * self.fps))
        self.stop_event = Event()

    def clear(self):
        self.frames.clear()

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, type, value, traceback):
        self.stop_recording()

    def start_recording(self):
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
        self.stop_event.set()
        self.thread.join()
        self.camera.release()

    def _record(self):
        while not self.stop_event.is_set():
            grabbed, frame = self.camera.read()
            if grabbed:
                self.frames.append(frame)

    def get_last_frames(self, seconds=None):
        if seconds is None:
            return list(self.frames)
        else:
            return list(self.frames)[-int(self.fps * seconds):]

    def capture(self):
        while True:
            if len(self.frames) > 0:
                return self.frames[-1]
