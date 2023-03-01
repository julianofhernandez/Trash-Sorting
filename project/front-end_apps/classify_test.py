import classify
#import settings
#from classify import camera_classify

import unittest
import cv2
from ssd import ssd_preds
from camera import *

class TestFileClassify(unittest.TestCase):

#def test_camera_classify_preds_none():
#    classify.main(True, True, 10)
    #test if user enters 0
    #C:\Users\santi\Downloads\dasani.jpg
    #input_file = input()
#    img = cv2.imread(input_file)
#    ssd_preds(img, True, True) == None
#    assert str == "Failed to classify"


#@pytest.fixture
#def fruit_bowl():
#    return [Fruit("apple"), Fruit("banana")]

    def test_file_classify(self):
        #Test the function when an invalid image path is entered
        #with self.assertRaises(TypeError):
            #classify.file_classify(True, True)

        # Test the function when a valid image path is entered
        input_file = "test_image.jpg"
        img = cv2.imread(input_file)
        expected_output = [('car', 0.99), ('bus', 0.88), ('truck', 0.75)]
        actual_output = ssd_preds(img, True, True)
        self.assertEqual(actual_output, expected_output)

    def test_file_classify_image_none():
        #classify.camera_classify(True, True)
        #classify.main(True, True, 1)
        #test if user enters 1
        img = cv2.imread(None)
        #assert classify.file_classify(True, True) == "classifying from file"
        #assert classify.file_classify(True, True) == "classifying from file"
        classify.file_classify(True, True)
        input_file = None
        img = None
        #assert classify.camera_classify(True, True) == "classifying from camera"
        #assert str == "Starting up camera..."
        #img = cv2.imread(None)
        assert str == "Image could not be read"

    def test_file_classify_preds_none():
        classify.main(True, True, 1)
        #test if user enters 1
        input_file = input()
        img = cv2.imread(input_file)
        ssd_preds(img, True, True) == None
        assert str == "Failed to classify"

    def test_file_classify_valid():
        classify.main(True, True, 1)
        #test if user enters 1
        input_file = input()
        img = cv2.imread(input_file)
        ssd_preds(img, True, True)
        #assert str == ?????

    def test_real_time_classify_preds_none():
        classify.main(True, True, 1)
        #test if user enters 2
        fps_rate = 30
        with CameraRecorder(1, fps=fps_rate) as cr:
            img = cr.capture()
        ssd_preds(img, True, True) == None
        assert str == "Failed to classify"

    def test_real_time_classify_valid():
        classify.main(True, True, 1)
        #test if user enters 2
        fps_rate = 30
        with CameraRecorder(1, fps=fps_rate) as cr:
            img = cr.capture()
        ssd_preds(img, True, True)
        #assert str == ???

if __name__ == '__main__':
    unittest.main()