from utils import (
    rev_label_map, 
)
from utils import detect

import torch
import cv2
import numpy as np
import argparse
import os

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(rev_label_map), 3))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model checkpoint.
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
print(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', default='inference_data/image_1.jpg',
        help='path to the input image'
    )
    parser.add_argument(
        '-t', '--threshold', default=0.2,
        help='detection threshold below which detections are dropped'
    )
    parser.add_argument(
        '-mo', '--max-overlap', dest='max_overlap', default=0.5,
        help='NMS overlap'
    )
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = parse_opt()
    img_path = args['input']
    min_score = args['threshold']
    max_overlap = args['max_overlap']
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    result = detect(
        original_image, 
        min_score=min_score, 
        max_overlap=max_overlap, 
        top_k=200,
        device=device,
        model=model,
        colors=COLORS
    )
    # Output save file name.
    save_name = img_path.split(os.path.sep)[-1].split('.')[0]
    cv2.imwrite(
        os.path.join('outputs', save_name+'.png'),
        result
    )
    cv2.imshow('Image', result)
    cv2.waitKey(0)