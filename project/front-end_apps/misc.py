'''
misc.py
Provides a list of helper functions to be used by other classes
Currrently provides a list of functions for reading in different inputs

Last modified 11/17 by Daniel Smagly
'''

def read_input(valid_inputs):

	''' reads input and returns the first character of the input if
	the first character is part of a valid input. Returns -1 otherwise
	for invalid inputs
	'''

	action = input()

	if(len(action) <= 0 or action[0].upper() not in valid_inputs):
		return -1

	return action[0].upper()

def read_input_tokens(valid_inputs):

	''' reads input and tokenizes valid input. Returns -1 otherwise
	for invalid inputs
	'''

	action = input()

	if(len(action) <= 0):
		return -1

	action = action.split()

	if(action[0].upper() not in valid_inputs):
		return -1
	
	action[0] = action[0].upper()

	return action

def read_number(low, high):

	''' reads input and returns the input if it is a numeric value 
	within the range [low, high]
	'''

	action = input()

	if(len(action) <= 0 or not action.isdigit()):
		return -1

	action = int(action)

	if(action < low or action > high):
		return -1

	return action

def print_invalid_input():
	print("ERROR: invalid option\n")

def print_menu_return():
	print("returning to menu")

def return_to_menu():
	main.main()

if __name__ == "__main__":
    pass
