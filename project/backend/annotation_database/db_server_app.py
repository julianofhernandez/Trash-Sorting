"""
The REST API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.
This flask app is deprecated in favor of the Consolidated Server, but
it may be run individually for alternative purposes.
"""

from flask import Flask
import argparse


# Define a function to print messages with a [Server] prefix for logging purposes.
def display(data):
    print("[Server]: " + data)


# Display server initialization message.
display("Attempting to initialize the server...")


def create_annotation_server():
    """
    Create and configure the Flask application and register the db_server blueprint.
    """
    # Import necessary components from the db_server module.
    from db_server import db_server, setup, IMAGE_DIR

    app = Flask(__name__)

    # Configure the upload folder and maximum content length for file uploads.
    app.config["UPLOAD_FOLDER"] = IMAGE_DIR
    app.config["MAX_CONTENT_LENGTH"] = (
        128 * 1024 * 1024
    )  # Set maximum file size to 128 MB.

    # Set up the database for the server.
    setup()

    # Register the db_server blueprint with the root URL prefix.
    app.register_blueprint(db_server, url_prefix="")
    return app


def run_server(host, port):
    """
    Create the annotation server and run it using the provided host and port.
    """
    # Create the Flask application instance.
    app = create_annotation_server()

    # Display a message indicating successful server launch.
    display("Successfully launched server")

    # Run the server with debug=False, threaded=False, and provided host and port.
    app.run(debug=False, threaded=False, host=host, port=port)


if __name__ == "__main__":
    # Set up a parser for command-line arguments.
    parser = argparse.ArgumentParser(
        prog="Annotation Database Server",
        description="Server that can send and receive images and annotations.",
    )

    # Add command-line arguments for host and port.
    parser.add_argument(
        "--host", default="localhost", help="Host address to bind server to."
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port number to bind server to."
    )

    # Parse command-line arguments.
    args = parser.parse_args()

    # Run the server with the parsed host and port arguments.
    run_server(args.host, args.port)
