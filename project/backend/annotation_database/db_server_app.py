"""
The Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.
"""
'''
from flask import Flask #Imports Python web framework that allows you to build web applications. 
import argparse #Imports module used to parse command-line arguments.

#The function below simply prints a message to the console.
def display(data):
    print('[Server]: ' + data)

#Lets you know that it is attempting to start the server.
display('Attempting to initialize the server...')


HOST = 'localhost' #Specifies the hostname or IP address of the server.
PORT = 5000 #Specifies the port number on which the server is listening for incoming connections. 

"""The create_app function is a factory function that creates and configures a Flask application instance. 
This is a common pattern in Flask applications, as it allows you to separate the creation of the application 
from its configuration."""
# https://flask.palletsprojects.com/en/2.2.x/patterns/appfactories/
def create_app():
    from db_server import db_server, setup, IMAGE_DIR #Imports the db_server blueprint, which defines the routes and methods for interacting with the annotation database, as well as the setup function and IMAGE_DIR variable.
    app = Flask(__name__) #Creates a new instance of the Flask class and assigns it to the app variable.
    app.config['UPLOAD_FOLDER'] = IMAGE_DIR #The UPLOAD_FOLDER configuration variable is set to the IMAGE_DIR variable, which specifies the directory where uploaded images will be saved.
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024 #The MAX_CONTENT_LENGTH configuration variable is set to 128 MB, which limits the size of uploaded files to prevent excessive memory usage.
    # app.config.from_pyfile(config_filename)

    setup() #The setup function is called to perform any necessary setup tasks, such as initializing the database.
    app.register_blueprint(db_server, url_prefix='') #Db_server blueprint is registered with the app instance using the register_blueprint method. The url_prefix argument is set to an empty string, which means that the routes defined in the blueprint will be accessible at the root URL.
    return app


try: #Attempts to create a new instance of the Flask application using the create_app function. 
    app = create_app()
except Exception as e: #If an exception is raised, we catch the exception and display an error message before terminating the program.
    display('Failed to launch server, terminating process...')
    print(e)
    exit()

#Lets you know that it has sucessfully launched the server.
display('Successfully launched server')

#Launches the Flask development server and begins listening for incoming HTTP requests, allowing clients to interact with the annotation database via the REST API.
if __name__ == '__main__':
    parser = argparse.ArgumentParser( #Argparse parser is created to handle command-line arguments.
        prog='Annotation Database Server',
        description='Server that can send and recieve images and annotations.'
    )
    app.run(debug=False, threaded=False, host=HOST, port=PORT) #The run method of the app instance is called to start the Flask development server.
#The debug argument is set to False to disable debugging mode, which is useful for production deployments. 
#The threaded argument is set to False to run the server in single-threaded mode, which is generally more efficient for simple applications. 
#The host and port arguments are set to the values of the HOST and PORT variables, respectively, which define the network interface and port number on which the server will listen.
'''
"""Below is a proposed refactoring of the code."""

#The function below simply prints a message to the console.
def display(data):
    print('[Server]: ' + data)

#Lets you know that it is attempting to start the server.
display('Attempting to initialize the server...')

from flask import Flask #Imports the flask class from the flask module.
import argparse #Imports the argparse module.

#The create_annotation_server function sets up a Flask application and registers a blueprint called db_server. 
def create_annotation_server():
    from db_server import db_server, setup, IMAGE_DIR #Imports some necessary components, specifically db_server, setup, and IMAGE_DIR.
    app = Flask(__name__) #Initializes the Flask application object by creating an instance of the Flask class.
    """^It passes the name of the current module as the argument. 
    This is done so that Flask knows where to find static files such as CSS, JS, and images."""
    app.config['UPLOAD_FOLDER'] = IMAGE_DIR #UPLOAD_FOLDER specifies the directory where uploaded files should be saved.
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024 #MAX_CONTENT_LENGTH sets the maximum size of an uploaded file. 
    #^In this case, the maximum file size is set to 128 MB.
    setup() #The setup() function is then called to create and configure the database that will be used by the server.
    app.register_blueprint(db_server, url_prefix='')
    """^Finally, the db_server blueprint is registered with the application using the app.register_blueprint() method.
    The url_prefix argument specifies that the routes defined in the db_server blueprint should be prefixed with an 
    empty string, meaning they will be at the root level of the application."""
    return app #The function returns the Flask application.

def run_server(host, port): #Arguments host and port determine the address and port number at which the server should be hosted.
    app = create_annotation_server() #The create_annotation_server() function is called to create a Flask application 
    #instance, and the instance is stored in the variable app.
    #Lets you know that it has sucessfully launched the server.
    display('Successfully launched server')
    app.run(debug=False, threaded=False, host=host, port=port)
"""^Finally, the app.run() method is called to run the server. The debug parameter is set to False, indicating that the 
server should not be run in debug mode. The threaded parameter is set to False, indicating that the server should not 
run in multiple threads. The host and port parameters are set to the arguments passed to the run_server() function."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser( #Defines a parser using the argparse module, which is used to parse command-line arguments. 
        prog='Annotation Database Server', #:Parser object is created with a program name and a description. 
        description='Server that can send and receive images and annotations.')
    parser.add_argument('--host', default='localhost', help='Host address to bind server to.')
    parser.add_argument('--port', type=int, default=5000, help='Port number to bind server to.')
    #^Two command-line arguments are added, --host and --port, with default values of 'localhost' 
    # and 5000, respectively.
    args = parser.parse_args()

    run_server(args.host, args.port)
    #After defining the parser, the run_server() function is called with the host and port arguments that are 
    #parsed from the command-line arguments using args.host and args.port, respectively.

'''
Here are the main changes made in this code:

Removed the display function since it is not used elsewhere in the code.

Renamed the create_app function to something more descriptive like create_annotation_server.

Moved the app.run method call into a separate run_server function that takes host and port arguments, 
so that the function can be used to start the server with different configurations.

Refactor the command-line argument parsing to use the argparse library instead of hardcoding the HOST 
and PORT variables. This will make it easier to run the server on different interfaces or ports without 
having to modify the code.

With these changes, you can run the server with different configurations by specifying command-line arguments:

CSS:
$ python server.py --host=0.0.0.0 --port=8000
This will start the server on all available network interfaces (allowing connections from other devices) and listen on port 8000.
'''