'''
about.py
This gives the about informational section of the program

Last modified 11/2 by Jeff de Jesus
'''

about_info = """
This program is intended to classify trash 
types in order to determine which trash bin 
it belongs in. Classifications are done 
through the use of a computer's webcam, and 
our program has been tested to be at least 
75% accurate with it's results."""

def main():
	print("About")
	print(about_info + "\n")

if __name__ == "__main__":
    main()