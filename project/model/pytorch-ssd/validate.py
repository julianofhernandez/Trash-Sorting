"""
Calculates COCO mAP on the validation set.

USAGE:
python validate.py
"""

from datasets import create_valid_dataset, create_valid_loader
from tqdm import tqdm
from pprint import PrettyPrinter
from utils import voc_labels as classes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b', '--batch-size', dest='batch_size', default=1,
    type=int, help='batch size for training and validation'
)
parser.add_argument(
    '-j', '--workers', default=4, type=int, 
    help='number of parallel workers'
)
parser.add_argument(
    '-d', '-data-dir', dest='data_dir', default='VOCdevkit',
    help='path to the VOCdevkit directory'
)
args = vars(parser.parse_args())

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = args['data_dir']
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = args['batch_size']
workers = args['workers']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = './checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = create_valid_dataset(
    data_folder=data_folder,
    train=False,
    keep_difficult=keep_difficult,
    resize_width=300,
    resize_height=300,
    use_train_aug=False,
    classes=list(classes),
)
print(f'Validation dataset has {len(test_dataset)} images')
test_loader = create_valid_loader(
    valid_dataset=test_dataset,
    batch_size=batch_size,
    num_workers=workers
)

metric = MeanAveragePrecision(class_metrics=True)

def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    target = list()
    preds = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            # true_dict = dict()
            # preds_dict = dict()
            # if i == 1:
            #     break
            images = images.to(device)  # (N, 3, 300, 300)
            orig_h = images.shape[2]
            orig_w = images.shape[3]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            for j, batch_data in enumerate(det_boxes_batch):
                true_dict = dict()
                preds_dict = dict()
                # for box in boxes:
                true_dict['boxes'] = boxes[j]
                # for label in labels:
                true_dict['labels'] = labels[j] # Needs to be a list for torchmetrics.
                target.append(true_dict)
    
                # for det_box in det_boxes_batch:
                preds_dict['boxes'] = det_boxes_batch[j]
                # for det_score in det_scores_batch:
                preds_dict['scores'] = det_scores_batch[j] # Needs to be a list for torchmetrics.
                # for det_label in det_labels_batch:
                preds_dict['labels'] = det_labels_batch[j] # Needs to be a list for torchmetrics.
                preds.append(preds_dict)

        metric.update(preds, target)
        pprint(metric.compute())

if __name__ == '__main__':
    evaluate(test_loader, model)