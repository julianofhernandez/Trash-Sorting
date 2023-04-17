"""
Automated Tests for Annotate.py
There are manual test that can be ran in annotate_test.ipynb
which test the UI since the UI cannot have automated tests.
"""

import annotate
import cv2
import os
import numpy as np

temp_path = "temp_img.png"


def get_img():
    if not os.path.isfile(temp_path):
        from matplotlib import pyplot as plt

        plt.plot([1, 2, 3], [1, 2, 3])
        plt.savefig(temp_path)

    return temp_path


def del_img():
    if os.path.isfile(temp_path):
        os.remove(temp_path)


class TestOpenFromPath:
    def test_invalid_path(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "fake_image.jpg")

        assert None == annotate.open_from_path()

    def test_valid_path(self, monkeypatch):
        get_img()

        monkeypatch.setattr("builtins.input", lambda _: temp_path)
        img = annotate.open_from_path()
        assert isinstance(img, np.ndarray)

        del_img()


class TestUploadAnnotation:
    def test_finished_annotation(self, capsys):
        img = cv2.imread(get_img())
        annotation = annotate.Annotations(None)

        annotation.finish()

        annotate.upload_annotation(annotation, img)

        del_img()

        stdout, stderr = capsys.readouterr()
        assert stdout == "[]\n"
        assert stderr == ""

    def test_unfinished_annotation(self, capsys):
        img = cv2.imread(get_img())
        annotation = annotate.Annotations(None)

        annotate.upload_annotation(annotation, img)

        del_img()

        stdout, stderr = capsys.readouterr()
        assert stdout == "Exited Annotation UI\n"
        assert stderr == ""
