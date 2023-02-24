import os
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import argparse



from model import SSD300, MultiBoxLoss
from datasets import (
    create_train_dataset, 
    create_train_loader,
)
from utils import (
    adjust_learning_rate, 
    AverageMeter,
    save_checkpoint,
    clip_gradient,
    label_map,
    taco_labels as classes
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b', '--batch-size', dest='batch_size', default=1,
    type=int, help='batch size for training and validation'
)
parser.add_argument(
    '-i', '--iterations', default=1200, type=int,
    help='iterations to train for'
)
parser.add_argument(
    '-j', '--workers', default=4, type=int, 
    help='number of parallel workers'
)
parser.add_argument(
    '-pf', '--print-frequency', dest='print_frequency', default=3,
    type=int, help='iteration interval for terminal log'
)
parser.add_argument(
    '-lr', '--learning-rate', dest='learning_rate', default=1e-4,
    type=float, help='default learning rate'
)
parser.add_argument(
    '-ckpt', '--checkpoint', default=None, type=str,
    help='path to trained checkpoint'
)
parser.add_argument(
    '-d', '--data-dir', dest='data_dir', default='TACO',
    help='path to the TACO directory'
)
args = vars(parser.parse_args())

# Data parameters
data_folder = args['data_dir']  # folder with data files
keep_difficult = False  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = args['batch_size']  # batch size
iterations = args['iterations']  # number of iterations to train
workers = args['workers']  # number of workers for loading data in the DataLoader
print_freq = args['print_frequency']  # print training status every __ batches
lr = args['learning_rate']  # learning rate
#######decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_at = [750, 1000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.85  # momentum
########weight_decay = 4e-4  # weight decay
weight_decay = 4e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)

        print(model)

        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = create_train_dataset(
        data_folder=data_folder,
        train=True,
        keep_difficult=keep_difficult,
        resize_width=300,
        resize_height=300,
        use_train_aug=False,
        classes=list(classes)
    )
    print(f'Training dataset has {len(train_dataset)} images')
    train_loader = create_train_loader(
        train_dataset=train_dataset,
        batch_size=batch_size,
        num_workers=workers,   
    )

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs).
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch.
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations.

    #epochs = iterations // (len(train_dataset) // 32)

    epochs = 5
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
    print(f"Training for {epochs} epochs")

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:

            #acc_score = accuracy_score(model=model, train_loader=train_loader, device=device)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss val: {loss.val:.25f} Avg loss: ({loss.avg:.25f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

"""def accuracy_score(model, train_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, boxes, labels, _ in train_loader:
            images = images.to(device)
            
            labels = [l.to(device) for l in labels]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total"""


if __name__ == '__main__':
    main()