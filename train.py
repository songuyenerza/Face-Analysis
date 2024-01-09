import argparse
import logging
import os
from threading import local
import time
import logging

import torch
import torch.utils.data as data

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch import distributed

import losses
from config import config as cfg
from dataset import  DataLoaderX, FaceDataset
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from backbones import get_model

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

torch.backends.cudnn.benchmark = True

class LSR2(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        balance_weight = torch.tensor([4.833, 1.607, 0.22977, 7.37, 2.7, 4.03]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        
#         one_hot += smooth_factor / length
        return one_hot.to(target.device)

    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def evaluate(model, val_loader, criterion, num_classes):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    # Prepare to track per-class metrics
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = val_loss / len(val_loader)

    # Calculate metrics per class
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=np.arange(num_classes), average=None)

    # Calculating overall accuracy
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))

    # Return the average loss, overall accuracy, and per-class metrics
    return avg_loss, accuracy, precision, recall, f1

class ClassificationModel(nn.Module):
    def __init__(self, backbone, cfg):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.fp16 = cfg.fp16
        #   if want freeze all backbone
        self.freeze_backbone()  # Freeze the backbone
        #   ////////////////////////////
        # Replace 'num_classes' with the actual number of classes
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.num_classes)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        with torch.cuda.amp.autocast(self.fp16):
            out = self.classifier(features)
        output = out.float() if self.fp16 else out
        return output

def main():

    #   logging
    log_filename = cfg.log
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #   dataset
    print("------------------------------------------------------")
    class_dict = {'Teenager' : '0', '40-50s': '1', '20-30s': '2', 'Baby': '3', 'Kid': '4', 'Senior': '5'}
    data_balance_weight_np = [4.833, 1.607, 0.22977, 7.37, 2.7, 4.03]
    # data_balance_weight_np = [0.528, 1.587, 11.106, 0.346, 0.942, 0.633]

    #   train   //
    trainset = FaceDataset(root_dir=cfg.rec, json_path = "../../../data/face_cropped/age/data_age_train.json", dict_class = class_dict)

    train_loader = data.DataLoader(
        dataset=trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)
    
    #   val //

    valset = FaceDataset(root_dir=cfg.rec, json_path = "../../../data/face_cropped/age/data_age_val.json", dict_class = class_dict)

    val_loader = data.DataLoader(
        dataset=valset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)


    #   //////////////////

    #   Model
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    #   load backbone
    backbone_pretrained = "./pretrained/r50_Glint360k_backbone.pth"
    backbone.load_state_dict(torch.load(backbone_pretrained, map_location='cuda'))
    logging.info("[init model ok]")

    logging.info(f"[config] {cfg}")



    # print(backbone)
    # Create the full model
    model = ClassificationModel(backbone, cfg).cuda()
    print("[init model ClassificationModel]")
    logging.info("[init model ClassificationModel]")


    model.train()
    model.cuda()
    #   ////////////
    #   todo: sonnt     optimizer
    if cfg.optimizer == 'sgd':
        opt = torch.optim.SGD(
            params=[{'params': model.parameters()}],
            lr=cfg.lr / 512 * cfg.batch_size,
            momentum=0.9, weight_decay=cfg.weight_decay)
    #   todo: sonnt ...... >> add optimizer
    elif cfg.optimizer == 'adamw':
        opt = torch.optim.AdamW(
            params=[{'params': model.parameters()}],
            lr=cfg.lr / 512 * cfg.batch_size)    
    else:
        raise
    # Loss function
    class_weights = torch.FloatTensor(data_balance_weight_np).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #   ///////////////////

    #   Training
    # Training loop
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    best_val_accuracy = 0
    for epoch in range(cfg.num_epoch):  # Define num_epochs
        epoch_start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            # Forward pass
            outputs = model(inputs)

            # loss = criterion(outputs, smoothed_labels)
            loss = LSR2(0.3)(outputs, labels)
            
            # Backward and optimize
            if cfg.fp16 == True:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()

            opt.zero_grad()
            

            if (i+1) % 100 == 0:
                val_loss, val_accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion, cfg.num_classes)
                logging.info("------------------------------------------------------")
                logging.info(f'Epoch [{epoch+1}/{cfg.num_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                logging.info(f'Precision per class: {precision}')
                logging.info(f'Recall per class: {recall}')
                logging.info(f'F1 Score per class: {f1}')

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        time_est = epoch_time * (cfg.num_epoch - epoch) / 3600
        logging.info(f"Estimated time to finish : {time_est:.2f} hours")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save model checkpoint
            # save_path = os.path.join(cfg["save_path"], f'model_epoch_{epoch+1}.pth')
            save_path = os.path.join(cfg["save_path"], f'model.pth')
            save_model(model, opt, epoch, save_path)
            logging.info(f"Model saved: {save_path}")

if __name__ == "__main__":
    main()