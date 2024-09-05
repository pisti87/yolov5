# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda import amp  # for mixed precision
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import (
    check_dataset,
    increment_path,
    LOGGER,
    TQDM_BAR_FORMAT,
    check_img_size,
    colorstr
)
from utils.loss import ComputeLoss
from utils.torch_utils import select_device, EarlyStopping
from utils.dataloaders import create_dataloader
import val as validate

def parse_opt():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300, help='total number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size across all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train and test image sizes (min, max)')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='optimizer weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='number of warmup epochs')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1, help='initial bias learning rate during warmup')

    # GPU and general settings
    parser.add_argument('--device', default='', help='cuda device id (0 for first GPU, "cpu" for CPU)')
    parser.add_argument('--project', default='runs/train', help='directory to save training logs and weights')
    parser.add_argument('--name', default='exp', help='experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='allow existing project/name, do not increment')
    parser.add_argument('--workers', type=int, default=8, help='number of data loader workers')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience (number of epochs without improvement)')

    # Mixed precision
    parser.add_argument('--mixed-precision', action='store_true', help='use mixed precision training')
    
    return parser.parse_args()

def main(opt):
    # Initialize save directory
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (opt.save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = select_device(opt.device)
    if device.type == 'cpu':
        LOGGER.warning('Using CPU. Consider switching to GPU for better performance.')
    
    # Load dataset
    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict['train'], data_dict['val']
    
    # Load model and resize images if necessary
    imgsz = check_img_size(opt.img_size[0], 32)  # check image size divisibility
    model = attempt_load(opt.weights, map_location=device)
    
    # Create data loaders
    train_loader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, opt.workers, device, augment=True)
    val_loader = create_dataloader(val_path, imgsz, opt.batch_size, opt.workers, device, augment=False)[0]
    
    # Optimizer setup
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    # Scheduler setup (OneCycleLR with warmup)
    lf = lambda x: ((1 - x / opt.epochs) * (1.0 - 0.2)) + 0.2  # cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Mixed precision scaler
    scaler = amp.GradScaler(enabled=opt.mixed_precision)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=opt.patience)

    # Loss function
    compute_loss = ComputeLoss(model)

    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        mloss = torch.zeros(1, device=device)
        pbar = tqdm(train_loader, total=len(train_loader), bar_format=TQDM_BAR_FORMAT, desc=f"Epoch {epoch+1}/{opt.epochs}")

        for i, (imgs, targets, paths, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # Normalize
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with amp.autocast(enabled=opt.mixed_precision):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets)
            
            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Mean loss computation
            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_description(f'Epoch [{epoch+1}/{opt.epochs}] Loss: {mloss.item():.4f}')
        
        # Scheduler step
        scheduler.step()
        
        # Validation and saving best model
        results, maps = validate.run(val_loader, model, device, imgsz)
        LOGGER.info(f'Validation results: {results}, mAP: {maps}')

        # Check early stopping criteria
        early_stopping(results[0], model, opt.save_dir / 'weights' / 'best.pt')
        if early_stopping.early_stop:
            LOGGER.info("Early stopping triggered")
            break

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
