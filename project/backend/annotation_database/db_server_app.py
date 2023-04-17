"""
The Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.
This flask app is Deprecated in favor of the Consolidated Server but
may be ran indivudually for alternative purposes
"""

from flask import Flask 
import argparse

def display(data):
    print('[Server]: ' + data)

display('Attempting to initialize the server...')

def create_annotation_server():
    """
    The create_annotation_server function sets up a Flask application and registers a blueprint called db_server. 

    Initializes the Flask application object by creating an instance of the Flask class.
    It passes the name of the current module as the argument. 
    This is done so that Flask knows where to find static files such as CSS, JS, and images.
    """
    from db_server import db_server, setup, IMAGE_DIR #Imports some necessary components, specifically db_server, setup, and IMAGE_DIR.
    app = Flask(__name__) 

    app.config['UPLOAD_FOLDER'] = IMAGE_DIR #UPLOAD_FOLDER specifies the directory where uploaded files should be saved.
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024 
    """ MAX_CONTENT_LENGTH sets the maximum size of an uploaded file. 
    In this case, the maximum file size is set to 128 MB.
    """

    # Create and configure the database that will be used by the server.
    setup() 

    """
    Finally, the db_server blueprint is registered with the application using the app.register_blueprint() method.
    The url_prefix argument specifies that the routes defined in the db_server blueprint should be prefixed with an 
    empty string, meaning they will be at the root level of the application.
    """
    app.register_blueprint(db_server, url_prefix='')

    return app

def run_server(host, port): 
    """
    Arguments host and port determine the address and port number at which the server should be hosted.
    """

    app = create_annotation_server() 
    """
    The create_annotation_server() function is called to create a Flask application 
    instance, and the instance is stored in the variable app.
    """

    display('Successfully launched server')

    """
    Finally, the app.run() method is called to run the server. The debug parameter is set to False, indicating that the 
    server should not be run in debug mode. The threaded parameter is set to False, indicating that the server should not 
    run in multiple threads. The host and port parameters are set to the arguments passed to the run_server() function.
    """
    app.run(debug=False, threaded=False, host=host, port=port)

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

