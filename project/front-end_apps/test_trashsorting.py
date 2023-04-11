import os
import pytest
import subprocess
import json

class TestTrashSortingApp:

    def test_input_image(self):
        # test using an input image
        result = subprocess.run(['python', 'trashsorting.py', '-i', r'..\..\images\validation\Organic Waste\O_12376.jpg'], stdout=subprocess.PIPE)
        returnJson = str(result.stdout.decode())
        assert result.returncode == 0
        assert returnJson.splitlines()[0] == "{'error_code': 0,"
        result = subprocess.run(['python', 'trashsorting.py', '-i', 'notanimage.jpg'], stdout=subprocess.PIPE)
        returnJson = str(result.stdout.decode())
        assert result.returncode == 0
        assert returnJson.splitlines()[0] == "Image could not be read"

    def test_local_model(self):
        # test using a local model
        result = subprocess.run(['python', 'trashsorting.py', '-i', r'..\..\images\validation\Organic Waste\O_12376.jpg', '-l', 'model.h5'], stdout=subprocess.PIPE)
        returnJson = str(result.stdout.decode())
        assert result.returncode == 0
        assert returnJson.splitlines()[0] == "{'error_code': 0,"

    def test_online_model(self):
        # test using an online model
        result = subprocess.run(['python', 'trashsorting.py', '-i', r'..\..\images\validation\Organic Waste\O_12376.jpg', '-o', 'http://localhost:5001'], stdout=subprocess.PIPE)
        returnJson = str(result.stdout.decode())
        assert result.returncode == 0
        print(returnJson)
        assert returnJson.splitlines()[0] == "{'error_code': 0,"

    def test_single_option(self):
        # test using the single option
        result = subprocess.run(['python', 'trashsorting.py', '-i', r'..\..\images\validation\Organic Waste\O_12376.jpg', '-s'], stdout=subprocess.PIPE)
        assert result.returncode == 0
        assert b'predictions' in result.stdout

    def test_output_json(self):
        # test outputting to JSON
        result = subprocess.run(['python', 'trashsorting.py', '-i', r'..\..\images\validation\Organic Waste\O_12376.jpg', '-j', 'output.json'], stdout=subprocess.PIPE)
        assert result.returncode == 0
        assert os.path.exists('output.json')
        with open('output.json') as f:
            data = json.load(f)
            assert data['error_code'] == 0
            assert data['predictions']['object_class'] == "Paper bag"
        os.remove('output.json')
