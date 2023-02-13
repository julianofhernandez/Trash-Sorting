from utils import (
    rev_label_map, 
)
from utils import detect

import torch
import cv2
import numpy as np
import argparse
import os
import time

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
        '-i', '--input', default='inference_data/video_1.mp4',
        help='path to the input video'
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
    video_path = args['input']
    min_score = args['threshold']
    max_overlap = args['max_overlap']

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Output save file name.
    save_name = video_path.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(os.path.join('outputs', save_name+'.mp4'), 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            result = detect(
                frame, 
                min_score=min_score, 
                max_overlap=max_overlap, 
                top_k=200,
                device=device,
                model=model,
                colors=COLORS
            )
            result = np.ascontiguousarray(result)
            end_time = time.time()
            
            # Get the current fps.
            fps = 1 / (end_time-start_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            cv2.putText(
                result,
                text=f"{fps:.1f} FPS",
                org=(10, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            out.write(result)
            cv2.imshow('Prediction', result)
            # Press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")