from calendar import prmonth
from importlib.resources import path
from logging import captureWarnings
from unittest import result
import cv2
import os

prompt = 'Classify \n 1. Open Camera and Capture \n 2. Upload Picture \n 3. Capture in Real Time \n 4. Quit \n\n'
def open_camera(capture=False):
    if capture == False: 
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(10)
            if key == 27: # exit on ESC
                break
        cv2.destroyWindow("preview")
        vc.release()
    if capture == True:
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        result, image = cam.read()
        if result:
            cv2.imshow('Captured', image)
            cv2.imwrite('Captured.png', image)
            cv2.waitKey(10)
            cv2.destroyWindow('Captured')
        else:
            print('No iamge detected...')
        

val = input(prompt)


while True:
    if int(val) == 1:
        print('opening camera...')
        open_camera(capture=True)
    if int(val) == 2:
        filepath = input('Enter your file path \n')
        os.system('start ' + filepath)
    if int(val) == 3:
        print('caputuring live...')
        open_camera(capture=False)
    if int(val) == 4:
        print('quitting....')
        break
    val = input(prompt)
