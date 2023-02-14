import json
import requests
from io import BytesIO
import numpy as np

class TestCreate:
    def test_invalid_key(self):
        res = requests.post('http://localhost:5001/create/model/apple', 
            data={'key': 'wrongkey', 'metadata': json.dumps({'a': 5})}, 
            files={'model': open('README.md','rb')})
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 1,\\n  "error_msg": "Invalid Key",\\n  "successful": false\\n}\\n\'')

    def test_already_exists(self):
        res = requests.post('http://localhost:5001/create/model/apple', 
            data={'key': 'secretkey', 'metadata': json.dumps({'a': 5})}, 
            files={'model': open('README.md','rb')})
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 5,\\n  "error_msg": "Model already exists",\\n  "successful": false\\n}\\n\'')

    def test_missing_model(self):
        res = requests.post('http://localhost:5001/create/model/win', 
            data={'key': 'secretkey', 'metadata': json.dumps({'a': 5})}, 
            files={})
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 3,\\n  "error_msg": "Missing model in files part of request",\\n  "successful": false\\n}\\n\'')

    def test_missing_metadata(self):
        res = requests.post('http://localhost:5001/create/model/win', 
            data={'key': 'secretkey'},
            files={'model': open('README.md','rb')})
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 4,\\n  "error_msg": "Missing metadata in form part of request",\\n  "successful": false\\n}\\n\'')

    def test_new_success(self):
        res = requests.post('http://localhost:5001/create/model/win', 
            data={'key': 'secretkey', 'metadata': json.dumps({'b': 6})}, 
            files={'model': open('README.md','rb')})
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 0,\\n  "error_msg": null,\\n  "successful": true\\n}\\n\'')


class TestReadInference:
    def test_no_image(self):
        res = requests.post('http://localhost:5001/read/inference/apple', files={})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 3,\\n  "error_msg": "Missing image in files part of request",\\n  "model_name": "apple",\\n  "predictions": []\\n}\\n\'')

    def test_exception(self):
        #TODO rewrite after fully implementing ssd.py
        byte_arr = BytesIO()
        byte_arr.write(np.random.randint(0, 127, 10, dtype=np.uint8).tobytes())
        byte_arr.seek(0)
        res = requests.post('http://localhost:5001/read/inference/apple', files={'image': byte_arr})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 2,\\n  "error_msg": "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices",\\n  "model_name": "apple",\\n  "predictions": []\\n}\\n\'')
        
    def test_success(self):
        #TODO write after fully implementing ssd.py
        pass


class TestReadBatchInf:
    def test_missing_images(self):
        res = requests.post('http://localhost:5001/read/batch-inference/apple', data={'num_image': 2}, files={})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "batch_predictions": [],\\n  "error_code": 3,\\n  "error_msg": "Missing image_0 in files part of request",\\n  "model_name": "apple"\\n}\\n\'')

    def test_missing_num_image(self):
        byte_arr, byte_arr2 = '', ''
        res = requests.post('http://localhost:5001/read/batch-inference/apple', data={}, files={'image_0': byte_arr, 'image_1': byte_arr2})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "batch_predictions": [],\\n  "error_code": 4,\\n  "error_msg": "Missing num_image in form part of request",\\n  "model_name": "apple"\\n}\\n\'')

    def test_exception(self):
        #TODO rewrite after fully implementing ssd.py
        byte_arr = BytesIO()
        byte_arr.write(np.random.randint(0, 127, 10, dtype=np.uint8).tobytes())
        byte_arr.seek(0)
        byte_arr2 = BytesIO()
        byte_arr2.write(np.random.randint(0, 127, 10, dtype=np.uint8).tobytes())
        byte_arr2.seek(0)

        res = requests.post('http://localhost:5001/read/batch-inference/apple', data={'num_image': 2}, files={'image_0': byte_arr, 'image_1': byte_arr2})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "batch_predictions": [],\\n  "error_code": 2,\\n  "error_msg": "invalid index to scalar variable.",\\n  "model_name": "apple"\\n}\\n\'')

    def test_success(self):
        #TODO write after fully implementing ssd.py
        pass


class TestReadModelList:
    def test_list(self):
        res = requests.get('http://localhost:5001/read/model/list')
        res, res.content, json.loads(res.content)['model_list']
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 0,\\n  "error_msg": null,\\n  "model_list": [\\n    {\\n      "metadata": "{\\\\"a\\\\": 5}",\\n      "model_name": "apple"\\n    },\\n    {\\n      "metadata": "{\\\\"b\\\\": 6}",\\n      "model_name": "win"\\n    }\\n  ]\\n}\\n\'')


class TestReadModel:
    def test_unknown(self):
        res = requests.get('http://localhost:5001/read/model/watermelon')
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 3,\\n  "error_msg": "Model Not Found"\\n}\\n\'')

    def test_known(self):
        res = requests.get('http://localhost:5001/read/model/apple')
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            "b'## Model Inference\\n\\nREST API Server\\n'")


class TestReadMetadata:
    def test_unknown(self):
        res = requests.get('http://localhost:5001/read/metadata/watermelon')
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 3,\\n  "error_msg": "Model Metadata Not Found",\\n  "metadata": ""\\n}\\n\'')

    def test_known(self):
        res = requests.get('http://localhost:5001/read/metadata/apple')
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 0,\\n  "error_msg": null,\\n  "metadata": "{\\\\"a\\\\": 5}"\\n}\\n\'')


class TestUpdate:
    def test_invalid_key(self):
        res = requests.put('http://localhost:5001/update/model/apple', data={'key': 'badkey', 'metadata': json.dumps({'a': 7})})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 1,\\n  "error_msg": "Invalid Key",\\n  "successful": false\\n}\\n\'')

    def test_model(self):
        res = requests.put('http://localhost:5001/update/model/win', data={'key': 'secretkey', 'model': open('README.md','rb')})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 0,\\n  "error_msg": null,\\n  "successful": true\\n}\\n\'')

    def test_metadata(self):
        res = requests.put('http://localhost:5001/update/model/apple', data={'key': 'secretkey', 'metadata': json.dumps({'a': 7})})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 0,\\n  "error_msg": null,\\n  "successful": true\\n}\\n\'')
        res = requests.put('http://localhost:5001/update/model/apple', data={'key': 'secretkey', 'metadata': json.dumps({'a': 5})})
        res, res.content


class TestDelete:
    def test_invalid_key(self):
        res = requests.delete('http://localhost:5001/delete/model/apple', data={'key': 'badkey'})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 1,\\n  "error_msg": "Invalid Key",\\n  "successful": false\\n}\\n\'')

    def test_success(self):
        res = requests.delete('http://localhost:5001/delete/model/win', data={'key': 'secretkey'})
        res, res.content
        assert (str(res), str(res.content)) == ('<Response [200]>', 
            'b\'{\\n  "error_code": 0,\\n  "error_msg": null,\\n  "successful": true\\n}\\n\'')
