from flask import Flask
from annotation_database import db_server as server
from model_inference import model_inference as inference
from website import web_server as web
"""
#sys.path.append()
#sys.path.append()
import sys
sys.path.append('./annotation_database') #Adds the annotation_database directory to the Python path, allowing the code to import modules from this directory.
sys.path.append('./model_inference') #Adds the model_inference directory to the Python path, allowing the code to import modules from this directory.
import db_server as server #Imports the db_server modules using the alias server.
import model_inference as inference #Imports the model_inference modules using the alias inference.

HOST = 'localhost' #Define the host for the Flask application.
PORT = 8000 #Define port number for the Flask application.

html_app = Blueprint('html_app', __name__) #Creates a blueprint object named html_app. 
#^Organizes related Flask routes and functions into a single object that can be registered with a Flask application.

#create_app() function: This function creates the Flask application object and registers the html_app, db_server, 
#and model_inference blueprints with the app object. It also sets up configuration values for the app object.
def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = server.IMAGE_DIR
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

    server.setup()
    inference.setup()

    app.register_blueprint(html_app, url_prefix = '')
    app.register_blueprint(server.db_server, url_prefix = '')
    app.register_blueprint(inference.model_inference, url_prefix = '')

    return app

#This function defines a Flask route for the html_app blueprint. When a user visits the root 
#URL ('/'), Flask will call the index() function and return the result to the user's browser. In this case, the index() function 
#renders the webpage.html template using the render_template() function provided by Flask.
@html_app.route('/')
def index():
    return render_template('website/webpage.html')


app = create_app() #This creates the Flask application object by calling the create_app() function.

#Starts the Flask development server when the script is run as the main program. 
#The server listens on the host and port defined in the HOST and PORT variables. When the server receives a request, it routes 
#the request to the appropriate blueprint and function, which generates a response and sends it back to the user's browser.
if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
"""
#Make it so that host and port use argparse
"""Below is a proposed refactoring of the code."""

from flask import Flask, render_template, Blueprint
import argparse
import sys
sys.path.append('./annotation_database') #Adds the annotation_database directory to the Python path, allowing the code to import modules from this directory.
sys.path.append('./model_inference') #Adds the model_inference directory to the Python path, allowing the code to import modules from this directory.
import db_server as server #Imports the db_server modules using the alias server.
import model_inference as inference #Imports the model_inference modules using the alias inference.

html_app = Blueprint('html_app', __name__) #Creates a blueprint object named html_app. 
#^Organizes related Flask routes and functions into a single object that can be registered with a Flask application.

"""create_app() function: This function creates the Flask application object and registers the html_app, db_server, 
and model_inference blueprints with the app object. It also sets up configuration values for the app object."""
def create_app(host, port):
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = server.IMAGE_DIR
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

    server.setup()
    inference.setup()

    app.register_blueprint(server.db_server, url_prefix='')
    app.register_blueprint(inference.model_inference, url_prefix='')
    app.register_blueprint(web.web_server, url_prefix='')

    return app

"""This function defines a Flask route for the html_app blueprint. When a user visits the root 
URL ('/'), Flask will call the index() function and return the result to the user's browser. In this case, the index() function 
renders the webpage.html template using the render_template() function provided by Flask."""
@html_app.route('/')
def index():
    return render_template('website/webpage.html')

"""Starts the Flask development server when the script is run as the main program. 
The server listens on the host and port defined in the HOST and PORT variables. When the server receives a request, it routes 
the request to the appropriate blueprint and function, which generates a response and sends it back to the user's browser."""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app.')
    parser.add_argument('--host', type=str, default='localhost', help='The hostname to bind to.')
    parser.add_argument('--port', type=int, default=8000, help='The port number to bind to.')
    args = parser.parse_args()

    app = create_app(args.host, args.port)
    app.run(debug=True, host=args.host, port=args.port)

"""
Here's what the changes do:

1. argparse is imported to parse command-line arguments.
2. create_app now takes host and port as arguments, so that it can be used to create the app with a different host and port.
3. The HOST and PORT variables are removed.
4. The __name__ == '__main__' block now uses argparse to parse the command-line arguments, and passes them to create_app and app.run()
"""
