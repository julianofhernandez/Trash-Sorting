from flask import Flask, render_template

app = Flask(__name__)

HOST = 'localhost'
PORT = 8000


@app.route('/')
def index():
    return render_template('webpage.html')


if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
