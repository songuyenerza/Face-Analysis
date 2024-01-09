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

from lr_scheduler import build_scheduler
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip, reduction='none')
    return flip_loss_l    


def generate_flip_grid(w, h):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

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
        balance_weight = torch.tensor([0.528, 1.587, 11.106, 0.346, 0.942, 0.633]).to(one_hot.device)
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

def evaluate(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, inputs_flip in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs, _ = model(inputs)
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
        self.fp16 = cfg.fp16
        #   if want freeze all backbone
        # self.freeze_backbone_custum()  # Freeze the backbone
        #   ////////////////////////////
        # Replace 'num_classes' with the actual number of classes
        self.head = nn.Linear(cfg.embedding_size, cfg.num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_backbone_custum(self):
        i = 0
        for param in self.backbone.parameters():
            i += 1
            if i < 236: #459 
                param.requires_grad = False

    def forward(self, x):
        features, cam = self.backbone(x)
        with torch.cuda.amp.autocast(self.fp16):
            out = self.head(features)

        output = out.float() if self.fp16 else out

        fc_weights = self.head.weight
        fc_weights = fc_weights.view(1, cfg.num_classes, cfg.embedding_size, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)

        # attention
        B, C, _, _ = cam.shape
        feat = cam.view(B, 1, C, 7, 7) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W

        return output, hm

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
    balance_weight_np = [0.528, 1.587, 11.106, 0.346, 0.942, 0.633]
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

    #   ///////////////////
    # Loss function
    criterion = nn.CrossEntropyLoss()

    #   Training
    # Training loop
    best_val_accuracy = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    for epoch in range(cfg.num_epoch):  # Define num_epochs
        epoch_start_time = time.time()
        for i, (inputs, labels, inputs_flip) in enumerate(train_loader):
            inputs, labels, inputs_flip = inputs.cuda(), labels.cuda(), inputs_flip.cuda()

            # Forward pass
            expression_output, hm = model(inputs)
            expression_output_flip, hm_flip = model(inputs_flip)

            grid_l = generate_flip_grid(7, 7).cuda(non_blocking=True)  
            flip_loss = ACLoss(hm, hm_flip, grid_l, expression_output)    #N*num_class*num_class*num_class

            flip_loss = flip_loss.mean(dim=-1).mean(dim=-1) #N*num_class
            balance_weight = torch.tensor(balance_weight_np).cuda().view(cfg.num_classes,1)
            flip_loss = torch.mm(flip_loss, balance_weight).squeeze()
            
            loss = LSR2(0.3)(expression_output, labels) + 0.1 * flip_loss.mean()
            
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

            # Backward and optimize
            opt.zero_grad()

            if (i+1) % 100 == 0:
                val_loss, val_accuracy = evaluate(model, val_loader, criterion)
                logging.info("------------------------------------------------------")
                logging.info(f'Epoch [{epoch+1}/{cfg.num_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        time_est = epoch_time * (cfg.num_epoch - epoch) / 60
        logging.info(f"Estimated time to finish : {time_est:.2f} minutes")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save model checkpoint
            # save_path = os.path.join(cfg["save_path"], f'model_epoch_{epoch+1}.pth')
            save_path = os.path.join(cfg["save_path"], f'model.pth')
            save_model(model, opt, epoch, save_path)
            logging.info(f"Model saved: {save_path}")

if __name__ == "__main__":
    main()
