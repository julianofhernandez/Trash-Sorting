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

def main(process_online, single_classification, fps_rate):
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
		elif(key == menu_options[3]):
			misc.return_to_menu()
		else:
			misc.print_menu_return()
			return False

def camera_classify(process_online, single_classification):
	#clear screen
	
	print("classifying from camera")
	input('Enter to capture')
	cc = CameraCapturer()
	img = cc.capture()
	# send img to Server or Local Model
	res = ssd_preds(img, process_online, single_classification)
	if res['error_code'] == 0:
		preds = res['predictions']
		for pred in preds:
			print(pred)
		#continue
	else:
		# error
		print("Failed to classify")
	del cc


def file_classify(process_online, single_classification):

	print("classifying from file")
	print("Enter image file path to be classifyed: ")
	input_file = input()
	img = cv2.imread(input_file)

	if img is None:
		print("Image could not be read")

		return 0

	# send img to Server or Local Model
	res = ssd_preds(img, process_online, single_classification)
	if res['error_code'] == 0:
		preds = res['predictions']
		for pred in preds:
			print(pred)
		#continue
	else:
		# error
		print("Failed to classify")

def real_time_classify(process_online, single_classification, fps_rate):
	print("classifying in real time")
	# add real time camera function

if __name__ == "__main__":
    main()