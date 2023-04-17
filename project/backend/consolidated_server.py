"""
The main Flask app for serving the web server and any api calls
"""

import argparse
from flask import Flask
from annotation_database import db_server as server
from model_inference import model_inference as inference
from website_backend import web_server as web

def create_app(host, port):
    """
    Creates a Flask app with the specified host and port and registers blueprints for the annotation database server,
    model inference server, and website backend server.

    Args:
        host: The hostname or IP address to bind the app to.
        port: The port to bind the app to.

    Returns:
        A Flask app instance.
    """

     # Creates a new Flask app instance with the specified static folder.
    app = Flask(__name__, static_folder='./website/build')

    # Set the app's configuration values for the upload folder and max content length.
    app.config['UPLOAD_FOLDER'] = server.IMAGE_DIR
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

    # Setup for the model_inference and annotation database which creates the directories
    # 
    server.setup()
    inference.setup()

    app.register_blueprint(server.db_server, url_prefix='')
    app.register_blueprint(inference.model_inference, url_prefix='')
    app.register_blueprint(web.web_server, url_prefix='')

    return app


"""
Starts the Flask development server when the script is run as the main program. 
The server listens on the host and port defined in the HOST and PORT variables. When the server receives a request, it routes 
the request to the appropriate blueprint and function, which generates a response and sends it back to the user's browser.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app.')
    parser.add_argument('--host', type=str, default='localhost',
                        help='The hostname to bind to.')
    parser.add_argument('--port', type=int, default=8000,
                        help='The port number to bind to.')
    args = parser.parse_args()

    app = create_app(args.host, args.port)
    app.run(debug=False, threaded=False, host=args.host, port=args.port)
