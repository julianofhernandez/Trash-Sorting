"""
The Rest API for model inference stuff.
"""
"""
from flask import Flask
from flask_cors import CORS
import argparse

#The display() function is defined to print out messages with a [Server] prefix for logging purposes.
def display(data):
    print('[Server]: ' + data)

display('Attempting to initialize the server...')

HOST = 'localhost' #Set default host.
PORT = 5001  # https://stackoverflow.com/a/72797062 #Set default port.

def create_app(): #Create a Flask application and register a blueprint for handling image prediction requests.
    from model_inference import model_inference, setup, MODELS_DIR #Imports the model_inference and MODELS_DIR variables from a model_inference module.
    app = Flask(__name__)

    setup() #Calls the setup() function.
    app.register_blueprint(model_inference, url_prefix='')
    return app

try: #A try block is used to create the Flask application and register the blueprint.
    app = create_app()
    CORS(app) #Create CORS header setup to allow request from the domain.
except Exception as e: #If there is an exception, the display() function is called to print out a message indicating that the server failed to launch.
    display('Failed to launch server, terminating process...')
    print(e)#The exception is printed to the console.
    exit()
    
display('Successfully launched server') #Message is printed to the console to indicate that the server has successfully launched.

#Finally, the server is run by calling the app.run() function with debug=False, threaded=False, and the host and port variables 
#as arguments. The parser object is defined but is not actually used in this code.
if __name__ == '__main__':
    #from werkzeug.middleware.profiler import ProfilerMiddleware
    # app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[
    #                                  5], profile_dir='./')
    parser = argparse.ArgumentParser(
        prog='Model Inference Server',
        description='Server that handles predictions on images'
    )
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
"""
#Make it so that host and port use argparse
"""Below is a proposed refactoring of the code."""

#The Rest API for model inference stuff.

from flask import Flask
from flask_cors import CORS
import argparse

#The display() function is defined to print out messages with a [Server] prefix for logging purposes.
def display(data):
    print('[Server]: ' + data)


display('Attempting to initialize the server...')


HOST = 'localhost' #Set default host.
PORT = 5001  # https://stackoverflow.com/a/72797062 #Set default port.


def create_app(): #Create a Flask application and register a blueprint for handling image prediction requests.
    from model_inference import model_inference, setup, MODELS_DIR #Imports the model_inference and MODELS_DIR variables from a model_inference module.
    app = Flask(__name__)

    setup()  #Calls the setup() function.
    app.register_blueprint(model_inference, url_prefix='')
    return app

try: #A try block is used to create the Flask application and register the blueprint.
    app = create_app()
    CORS(app)  # Create CORS header setup to allow request from the domain
except Exception as e: #If there is an exception, the display() function is called to print out a message indicating that the server failed to launch.
    display('Failed to launch server, terminating process...')
    print(e) #The exception is printed to the console.
    exit()

display('Successfully launched server')

#Finally, the server is run by calling the app.run() function with debug=False, threaded=False, 
# and the host and port variables as arguments.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Model Inference Server')
    parser.add_argument('--host', type=str, default='localhost', help='The hostname to bind to.')
    parser.add_argument('--port', type=int, default=5001, help='The port number to bind to.')
    args = parser.parse_args()
    
    app.run(debug=False, threaded=False, host=args.host, port=args.port)

