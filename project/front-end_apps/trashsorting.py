import argparse
import cv2
from ssd import ssd_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Trash Sorting Application',
        description='This application classifies trash in images.')

    parser.add_argument('image_filename')
    parser.add_argument('-l', '--local',
                        help='A path to a local model to use for inference')
    parser.add_argument('-o', '--online', default='http://127.0.0.1:5001/',
                        help='A url to the REST API that hosts the inference model')
    parser.add_argument('-s', '--single',
                        action='store_true',
                        help='Only classify a single object in the image')

    args = parser.parse_args()

    img = cv2.imread(args.image_filename)
    if img is None:
        print("Image could not be read")
    else:
        # send img to Server or Local Model
        res = ssd_preds(img, args.online is not None, args.single)
        if res['error_code'] == 0:
            preds = res['predictions']
            for pred in preds:
                print(pred)
        else:
            # error
            print("Failed to classify")
