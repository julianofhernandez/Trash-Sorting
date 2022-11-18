'''
classify.py
the classify menu to be used to collect the data to be classified
by the user

Last modified 11/17 by Daniel Smagly
'''

import settings
import misc
import main
import camera
import cv2
from camera2 import *

menu_options = ['1', '2', '3','Q','M']

menu_prompt = menu_options[0] + ": Open Camera and capture\n" + \
	menu_options[1] + ": Upload picture\n" + \
	menu_options[2] + ": Capture real time\n" + \
	menu_options[3] + ": Exit capture\n" + \
	menu_options[4] + ": Return to menu"

def main():
	print("Classify trash")
	while(True):
		print(menu_prompt)
		print("Press the key to the corresponding action")

		key = misc.read_input(menu_options)
		
		if(key == -1):
			misc.print_invalid_input()
			continue

		if(key == menu_options[0]):
			camera_classify()
		elif(key == menu_options[1]):
			file_classify()
		elif(key == menu_options[2]):
			camera.main() 
		elif(key == menu_options[3]):
			misc.return_to_menu()
		else:
			misc.print_menu_return()
			return False

def camera_classify():
	#clear screen
	print("classifying from camera")
	input('Enter to capture')
	cc = CameraCapturer()
	img = cc.capture()
	# send img to Server
	res = ssd_preds(img)
	if res['error_code'] == 0:
		
		preds = res['predictions']
		for pred in preds:
			print(pred)
		#continue
	else:
		# error
		print("Failed to classify")
	del cc


def file_classify():
	print("classifying from file")
	print("Enter image file path to be classifyed: ")
	input_file = input()
	img = cv2.imread(input_file)

	if img is None:
		print("Image could not be read")
		return 0

	cv2.imshow("Entered image", img)
	k = cv2.waitKey(0)

	if k == ord("q"):
		cv2.imwrite("sample_img.png", img)
		misc.return_to_menu()
def real_time_classify():
	print("classifying in real time")
	# add real time camera function

if __name__ == "__main__":
    main()