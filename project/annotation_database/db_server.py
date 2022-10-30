def display(data):
	print("[Server]: " + data)
	
display("Attempting to initialize the server...")

from time import sleep
from flask import Flask, request, send_file
import logging
from document_scanner import scan

HOST = '192.168.1.199'
PORT = 5000

try:
	app = Flask(__name__)
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()
display("Successfully launched server")

#Creates a new entry
@app.route("/create/entry", methods = ['POST'])
def handle_entry():
	f = request.files['file']
	#to complete later...
	return None

#Creates many entries
@app.route("/create/entries", methods = ['POST'])
def handle_entries():
	f = request.files['file']
	#to complete later...
	return None

#Query for count of queries
@app.route("/query/count/<filter>", methods = ['GET'])
def handle_count_query(filter):
	return None

#Query for a single entry
@app.route("/query/entry/<filter>", methods = ['GET'])
def handle_get_entry(filter):
	return None

#Query for all entries given a single search
@app.route("/query/search/<filter>", methods = ['GET'])
def handle_search_entries(filter):
	return None

#Query for entry with no or least annotations
@app.route("/query/annotationless/<filter>", methods = ['GET'])
def handle_get_entry_no_annotation(filter):
	return None
	
#Query for entry with most annotations
@app.route("/query/maxannotation/<filter>", methods = ['GET'])
def handle_get_entry_max_annotations(filter):
	return None
	

				   
if __name__ == "__main__":
	app.run(threaded=True, host=HOST, port=PORT) 