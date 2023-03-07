import argparse
import cv2
import json
from ssd import ssd_preds
from camera import CameraRecorder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Trash Sorting Application',
        description='This application classifies trash in images.')

    # Define mutually exclusive group for input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input',
                       help='filename to an input image')
    group.add_argument('-c', '--camera',
                        action='store_true',
                        help='Opens camera and takes a picture')

    # Add other command-line options
    parser.add_argument('-l', '--local',
                        help='A path to a local model to use for inference')
    parser.add_argument('-o', '--online', default='http://127.0.0.1:5001/',
                        help='A url to the REST API that hosts the inference model')
    parser.add_argument('-s', '--single',
                        action='store_true',
                        help='Only classify a single object in the image')
    parser.add_argument('-j', '--json',
                        help='Saves a prediction output as json instead of going to console')

    args = parser.parse_args()
    

    if args.camera:
        with CameraRecorder(.5, fps=30) as cr:
            print("Active. Press Q to stop")
            while True:
                img = cr.capture()
                cv2.imshow("Classifing. Press Q to stop", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        cv2.waitKey(1) 
    else:
        img = cv2.imread(args.input)
    if img is None:
        print("Image could not be read")
    else:
        # Send the image to the Server or Local Model for classification
        preds = ssd_preds(img, args.online is not None, args.single)
        
        if preds is not None:
            if args.json:
                # If output is to be saved as json file, open a file and save the predictions
                with open(args.json, 'w') as f:
                    json.dump(preds, f)
            else:
                # If output is to be printed to console, iterate over predictions and print them
                for pred in preds:
                    print(pred)
        else:
            # error
            print("Failed to classify")