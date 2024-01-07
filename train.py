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

torch.backends.cudnn.benchmark = True

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def evaluate(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

class ClassificationModel(nn.Module):
    def __init__(self, backbone, cfg):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        #   if want freeze all backbone
        self.freeze_backbone()  # Freeze the backbone
        #   ////////////////////////////
        # Replace 'num_classes' with the actual number of classes
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg.num_classes)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out

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
        
    # Loss function
    criterion = nn.CrossEntropyLoss()

    #   Training
    # Training loop
    best_val_accuracy = 0
    for epoch in range(cfg.num_epoch):  # Define num_epochs
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i+1) % 100 == 0:
                val_loss, val_accuracy = evaluate(model, val_loader, criterion)
                logging.info(f'Epoch [{epoch+1}/{cfg.num_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                logging.info("------------------------------------------------------")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save model checkpoint
            # save_path = os.path.join(cfg["save_path"], f'model_epoch_{epoch+1}.pth')
            save_path = os.path.join(cfg["save_path"], f'model.pth')
            save_model(model, opt, epoch, save_path)
            logging.info(f"Model saved: {save_path}")

if __name__ == "__main__":
    main()