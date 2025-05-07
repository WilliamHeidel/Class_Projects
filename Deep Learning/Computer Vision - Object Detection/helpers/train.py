import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    #raise NotImplementedError('train')
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.learning_rate, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.patience)
    loss = torch.nn.BCEWithLogitsLoss()
    
    import ast
    brightness, contrast, saturation, hue =  ast.literal_eval(args.color_jitter)
    
    train_transforms = dense_transforms.Compose([
        dense_transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap()
    ])
    
    train_data = load_detection_data('dense_data/train', batch_size=args.batch_size, transform=train_transforms)
    valid_data = load_detection_data('dense_data/valid', batch_size=args.batch_size, transform=dense_transforms.Compose([dense_transforms.ToTensor(),dense_transforms.ToHeatmap()]))
    
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, peak, size in train_data:
            img, peak, size = img.to(device), peak.to(device), size.to(device)

            logit = model(img).to(device)
            loss_val = loss(logit, peak)
            #acc_val = accuracy(logit, label)
            
            train_logger.add_scalar('loss', loss_val, global_step=global_step)
            #.extend([acc_val.numpy()])
            log(train_logger, img, peak, logit, global_step)
            global_step += 1

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        #train_logger.add_scalar('accuracy', np.mean(acc_vals), global_step=global_step)
        train_logger.add_scalar('epoch', epoch+1, global_step=global_step)
        train_logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=global_step)

        model.eval()
        for img, peak, size in valid_data:
            #img, peak, size = img.to(device), peak.to(device), size.to(device)
            #logit = model(img).to(device)
            pass #vacc_vals.append(accuracy(model(img), label).numpy())
        #valid_logger.add_scalar('accuracy', np.mean(vacc_vals), global_step=global_step)
    
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-p', '--printout', type=bool, default=False)
    parser.add_argument('-pe', '--prob_exp', type=int, default=1)
    parser.add_argument('-op', '--optim', type=str, default='Adam')
    parser.add_argument('-es', '--early_stop', type=str, default="[0.90,0.6]")
    parser.add_argument('-cj', '--color_jitter', type=str, default="[(0.5,1.5),(0.5,1.5),(0.5,1.5),(-0.25,0.25)]")
    parser.add_argument('-pa', '--patience', type=int, default=5) 
    parser.add_argument('-sc', '--scheduler', type=str, default='acc')    
    parser.add_argument('-bs', '--batch_size', type=int, default=32) 

    args = parser.parse_args()
    train(args)
