'''
annotate.py
This is the file to open up the annotation gui which will add annotations
to images for training.
Last modified 11/3 by Jeff de Jesus
'''

import misc

# list of valid commands the user should be able to use in this menu
menu_options = ['1', '2', 'Q']

# the text prompt for this menu
menu_prompt = menu_options[0] + "\t: Opens GUI to capture a photo and annotate\n" + \
	menu_options[1] + " [URL] : Opens GUI and loads image from [URL] to annotaten\n" + \
	menu_options[2] + "\t: Exit Annotation"

def main(process_online, single_classification, fps_rate):
	while(True):
		print(menu_prompt)
		key = misc.read_input_tokens(menu_options)
		
		if(key == -1):
			misc.print_invalid_input()
			continue

		if(key[0] == menu_options[0]):
			open_annotation()
		elif(key[0] == menu_options[1]):
			open_annotation(url = key[1])
		else:
			misc.print_menu_return()
			return 0

def open_annotation(live_capture = False, url = ""):
	''' Function to open up the annotation gui as referenced in 
	https://github.com/julianofhernandez/Trash-Sorting/blob/main/images/Curated%20Diagrams.pdf
	live_capture 	will open it with the option of live capturing
	url		will open it with the image from the url loaded up
	'''
	print("url =", url)
	print("opening annotation window")

if __name__ == "__main__":
    main()