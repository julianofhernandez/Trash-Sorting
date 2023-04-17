import pytest
from model_inference_app import create_app
import json
import requests
import numpy as np
import cv2
import os


@pytest.fixture()
def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )

    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


@pytest.fixture()
def tmp_img(tmp_path):
    from matplotlib import pyplot as plt

    path = os.path.join(tmp_path, "temp_img.png")
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.savefig(path)

    return path


class TestCreate:
    def test_invalid_key(self, client):
        res = client.post(
            "http://localhost:5000/create/model/test",
            data={
                "key": "wrongkey",
                "metadata": json.dumps({"a": 5}),
                "model": open("README.md", "rb"),
            },
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_already_exists(self, client):
        res = client.post(
            "http://localhost:5000/create/model/test",
            data={
                "key": "secretkey",
                "metadata": json.dumps({"a": 5}),
                "model": open("README.md", "rb"),
            },
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 5,
            "error_msg": "Model already exists",
            "successful": False,
        }

    def test_missing_model(self, client):
        res = client.post(
            "http://localhost:5000/create/model/test2",
            data={"key": "secretkey", "metadata": json.dumps({"a": 5})},
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 3,
            "error_msg": "Missing model in files part of request",
            "successful": False,
        }

    def test_missing_metadata(self, client):
        res = client.post(
            "http://localhost:5000/create/model/test2",
            data={"key": "secretkey", "model": open("README.md", "rb")},
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 4,
            "error_msg": "Missing metadata in form part of request",
            "successful": False,
        }

    def test_new_success(self, client):
        res = client.post(
            "http://localhost:5000/create/model/test2",
            data={
                "key": "secretkey",
                "metadata": json.dumps({"b": 6}),
                "model": open("README.md", "rb"),
            },
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }


class TestReadInference:
    def test_no_image(self, client):
        res = client.post("http://localhost:5000/read/inference/test")
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 3,
            "error_msg": "Missing image in files part of request",
            "model_name": "test",
            "predictions": None,
        }

    def test_success(self, client, tmp_img):
        res = client.post(
            "http://localhost:5000/read/inference/test",
            data={"image": open(tmp_img, "rb")},
        )
        res_content = res.get_json()

        assert res.status_code == 200
        assert "error_code" in res_content and res_content["error_code"] == 0
        assert "error_msg" in res_content and res_content["error_msg"] is None
        assert "model_name" in res_content and res_content["model_name"] == "test"
        assert "predictions" in res_content
        assert set(
            [
                "object_class",
                "object_class_probs",
                "object_classes",
                "object_trash_class",
                "object_trash_class_probs",
                "trash_class",
                "trash_class_probs",
                "trash_classes",
            ]
        ).issubset(set(res_content["predictions"]))


class TestReadBatchInf:
    def test_missing_images(self, client):
        res = client.post(
            "http://localhost:5000/read/batch-inference/test", data={"num_image": 2}
        )
        res, res.get_json()
        assert res.status_code == 200
        assert res.get_json() == {
            "batch_predictions": None,
            "error_code": 3,
            "error_msg": "Missing image_0 in files part of request",
            "model_name": "test",
        }

    def test_missing_num_image(self, client, tmp_img):
        res = client.post(
            "http://localhost:5000/read/batch-inference/test",
            data={"image_0": open(tmp_img, "rb"), "image_1": open(tmp_img, "rb")},
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "batch_predictions": None,
            "error_code": 4,
            "error_msg": "Missing num_image in form part of request",
            "model_name": "test",
        }

    def test_success(self, client, tmp_img):
        res = client.post(
            "http://localhost:5000/read/inference/test",
            data={"image": open(tmp_img, "rb")},
        )
        res_content = res.get_json()

        print(res_content)

        assert res.status_code == 200
        assert "error_code" in res_content and res_content["error_code"] == 0
        assert "error_msg" in res_content and res_content["error_msg"] is None
        assert "model_name" in res_content and res_content["model_name"] == "test"
        assert "predictions" in res_content
        assert set(
            [
                "object_class",
                "object_class_probs",
                "object_classes",
                "object_trash_class",
                "object_trash_class_probs",
                "trash_class",
                "trash_class_probs",
                "trash_classes",
            ]
        ).issubset(set(res_content["predictions"]))


class TestReadModelList:
    def test_list(self, client):
        res = client.get("http://localhost:5000/read/model/list")
        assert res.status_code == 200
        rj = res.get_json()
        assert rj["error_code"] == 0
        assert rj["error_msg"] is None
        assert "model_list" in rj
        assert len(rj["model_list"]) == 7  # 6 but test2 is added temporarily


class TestReadModel:
    def test_unknown(self, client):
        res = client.get("http://localhost:5000/read/model/watermelon")
        assert res.status_code == 200
        assert res.get_json() == {"error_code": 3, "error_msg": "Model Not Found"}

    def test_known(self, client):
        res = client.get("http://localhost:5000/read/model/test")
        assert res.status_code == 200
        # assert str(res.get_json()) == "b'test'"


class TestReadMetadata:
    def test_unknown(self, client):
        res = client.get("http://localhost:5000/read/metadata/watermelon")
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 3,
            "error_msg": "Model Metadata Not Found",
            "metadata": "",
        }

    def test_known(self, client):
        res = client.get("http://localhost:5000/read/metadata/test")
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "metadata": json.dumps({"a": 7}),
        }


class TestUpdate:
    def test_invalid_key(self, client):
        res = client.put(
            "http://localhost:5000/update/model/test",
            data={"key": "badkey", "metadata": json.dumps({"a": 7})},
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_model(self, client):
        res = client.put(
            "http://localhost:5000/update/model/test2",
            data={"key": "secretkey", "model": open("README.md", "rb")},
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }

    def test_metadata(self, client):
        res = client.put(
            "http://localhost:5000/update/model/test",
            data={"key": "secretkey", "metadata": json.dumps({"a": 7})},
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }
        res = client.put(
            "http://localhost:5000/update/model/test",
            data={"key": "secretkey", "metadata": json.dumps({"a": 5})},
        )


class TestDelete:
    def test_invalid_key(self, client):
        res = client.delete(
            "http://localhost:5000/delete/model/test", data={"key": "badkey"}
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_success(self, client):
        res = client.delete(
            "http://localhost:5000/delete/model/test2", data={"key": "secretkey"}
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }
