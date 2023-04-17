"""
The Flask server for model inference purposes only.
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


# Define a function to create a Flask application and register a blueprint
# for handling image prediction requests.
def create_app():
    # Import model_inference and MODELS_DIR variables from the model_inference module.
    from model_inference import model_inference, setup, MODELS_DIR

    app = Flask(__name__)

    # Call the setup() function.
    setup()
    # Register the blueprint.
    app.register_blueprint(model_inference, url_prefix="")
    return app


# Attempt to create the Flask application and register the blueprint.
try:
    app = create_app()
except Exception as e:
    # Print an error message and terminate the process if server launch fails.
    display("Failed to launch server, terminating process...")
    print(e)
    exit()

# Display a message indicating successful server launch.
display("Successfully launched server")

# Run the server with debug=False, threaded=False,
# and provided host and port variables as arguments.
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run the Model Inference Server")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="The hostname to bind to."
    )
    parser.add_argument(
        "--port", type=int, default=5001, help="The port number to bind to."
    )
    args = parser.parse_args()

    # Run the Flask server.
    app.run(debug=False, threaded=False, host=args.host, port=args.port)
