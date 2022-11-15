'''
settings.py
where the settings of the program should be modified.
For now all the settings are not saved in file and
will be saved here

Last modified 11/3 by Jeff de Jesus
'''

import misc

process_online = False
single_classification = False
fps_rate = 1

MAX_FPS = 60
MIN_FPS = 1

menu_options = ['1', '2', '3','Q']
sub_options = ['1', '2']

menu_prompt = menu_options[0] + ": Toggle Online/Offline Computation\n" + \
	menu_options[1] + ": Toggle Single/Multi object Classification\n" + \
	menu_options[2] + ": Change FPS for live capture\n" + \
	menu_options[3] + ": Exit Settings"

def main():
	while(True):
		print(menu_prompt)
		print("Press the key to the corresponding action")

		key = misc.read_input(menu_options)
		
		if(key == -1):
			misc.print_invalid_input()
			continue

		if(key == menu_options[0]):
			toggle_computation_mode()
		elif(key == menu_options[1]):
			toggle_classification_mode()
		elif(key == menu_options[2]):
			toggle_fps()
		else:
			misc.print_menu_return()
			return 0
		

def toggle_computation_mode():
	''' Toggles the computation between online and offline
	'''
	print("Do you want online or offline computation?")
	print(sub_options[0] + ": online")
	print(sub_options[1] + ": offline")

	key = misc.read_input(sub_options)
	
	if(key == -1):
		misc.print_invalid_input()
		return -1

	if(key == sub_options[0]):
		process_online = True
		print("Processing is online")
	else:
		process_online = False
		print("Processing is offline")
		

def toggle_classification_mode():
	''' Toggles the classification mode between single and multi
	'''
	print("Do you want single or multi object classification?")
	print(sub_options[0] + ": single")
	print(sub_options[1] + ": multi")

	key = misc.read_input(sub_options)
	
	if(key == -1):
		misc.print_invalid_input()
		return -1

	if(key == sub_options[0]):
		single_classification = True
		print("classification is singular")
	else:
		single_classification = False
		print("classification is multi")

def toggle_fps():
	''' Adjusts the FPS rate for live capture
	''' 
	print("Enter a number for the framerate for live capture")
	print("Min framerate: ", MIN_FPS)
	print("Max framerate: ", MAX_FPS)
	
	key = misc.read_number(low = MIN_FPS, high = MAX_FPS)
	
	if (key == -1):
		misc.print_invalid_input()
		return -1

	fps_rate = key
	print("FPS is set to", fps_rate)
	
if __name__ == "__main__":
    main()