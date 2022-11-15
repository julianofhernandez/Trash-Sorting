'''
classify.py
the classify menu to be used to collect the data to be classified
by the user

Last modified 11/2 by Jeff de Jesus
'''

import settings
import misc

menu_options = ['1', '2', '3','Q']

menu_prompt = menu_options[0] + ": Open Camera and capture\n" + \
	menu_options[1] + ": Upload picture\n" + \
	menu_options[2] + ": Capture real time\n" + \
	menu_options[3] + ": Exit capture"

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
			real_time_classify()
		else:
			misc.print_menu_return()
			return 0

def camera_classify():
	print("classifying from camera")

def file_classify():
	print("classifying from file")

def real_time_classify():
	print("classifying in real time")

if __name__ == "__main__":
    main()