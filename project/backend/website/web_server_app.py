from flask import Flask
from web_server import web_server

HOST = 'localhost'
PORT = 8080


def create_app():
    app = Flask(__name__,  static_folder='build')

    app.register_blueprint(web_server, url_prefix='')
    return app


try:
    app = create_app()
except Exception as e:
    print(e)
    exit()


if __name__ == '__main__':
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
