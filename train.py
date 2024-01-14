import logging
import os
from threading import local
import time
import logging

import torch
import torch.utils.data as data

import torch.utils.data.distributed
import torch.nn as nn

from tqdm import tqdm
import shutil
from config import config as cfg
from dataset import  DataLoaderX, FaceDataset, FaceDatasetVal

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from backbones.face_feature_extractor import FeatureExtractor
from losses import LSR2
torch.backends.cudnn.benchmark = True


def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def evaluate(model, backbone, val_loader, criterion, num_classes):
    # model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    # Prepare to track per-class metrics
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            feature = backbone.forward(inputs)
            feature = torch.tensor(np.asarray(feature)).cuda()
            outputs = model(feature)
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
    def __init__(self, cfg):
        super(ClassificationModel, self).__init__()
        self.fp16 = cfg.fp16
        #   if want freeze all backbone
        #   ////////////////////////////
        # Replace 'num_classes' with the actual number of classes
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg.num_classes)
        )

    def forward(self, features):
        with torch.cuda.amp.autocast(self.fp16):
            out = self.classifier(features)
        output = out.float() if self.fp16 else out
        return output

def main():
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    shutil.copy('config.py', cfg.save_path)
    #   logging
    log_filename =   os.path.join(cfg.save_path, cfg.log)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #   dataset
    print("------------------------------------------------------")
    class_dict = cfg.class_dict
    data_balance_weight_np = cfg.data_balance_weight_np
    
    #   train   //
    trainset = FaceDataset(root_dir=cfg.rec, json_path = "../../../data/face_cropped/age/data_age_train.json", dict_class = class_dict)

    train_loader = data.DataLoader(
        dataset=trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=24, pin_memory=True, drop_last=True)

    #   ///////////////////////////////////////
    # train_sampler = DistributedSampler(
    #     trainset, num_replicas=1, rank=0, shuffle=True, seed=2048)
    # init_fn = partial(worker_init_fn, num_workers=16, rank=0, seed=2048)
    # train_loader = DataLoaderX(
    #     local_rank=0,
    #     dataset=trainset,
    #     batch_size=cfg.batch_size,
    #     sampler=train_sampler,
    #     num_workers=16,
    #     pin_memory=True,
    #     drop_last=True,
    #     worker_init_fn=init_fn,
    # )

    #   val //
    valset = FaceDatasetVal(root_dir=cfg.rec, json_path = "../../../data/face_cropped/age/data_age_val.json", dict_class = class_dict)

    val_loader = data.DataLoader(
        dataset=valset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)


    #   //////////////////

    #   Model

    backbone = FeatureExtractor(model_path='./pretrained/webface600_r50_fixoutput.onnx', 
                            useGpu=True, index_gpu=0)

    logging.info("[init model ok]")

    logging.info(f"[config] {cfg}")


    # Create the full model
    model = ClassificationModel(cfg).cuda()
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
    elif cfg.optimizer == 'adam':
        opt = torch.optim.Adam(
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
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs, labels = inputs.cuda(), labels.cuda()
            # Forward pass
            feature = backbone.forward(inputs)
            feature = torch.tensor(np.asarray(feature)).cuda()

            outputs = model(feature)

            loss = criterion(outputs, labels)
            # loss = LSR2(0.3, data_balance_weight_np)(outputs, labels)
            
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
                val_loss, val_accuracy, precision, recall, f1 = evaluate(model, backbone, val_loader, criterion, cfg.num_classes)
                logging.info("------------------------------------------------------")
                logging.info(f'Epoch [{epoch+1}/{cfg.num_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                logging.info(f'Precision per class: {precision}')
                logging.info(f'Recall per class: {recall}')
                logging.info(f'F1 Score per class: {f1}')

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        time_est = epoch_time * (cfg.num_epoch - epoch) / 3600
        logging.info(f"Estimated time to finish : {time_est:.2f} (hours)")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save model checkpoint
            # save_path = os.path.join(cfg["save_path"], f'model_epoch_{epoch+1}.pth')
            save_path = os.path.join(cfg["save_path"], f'model.pth')
            save_model(model, opt, epoch, save_path)
            logging.info(f"Model saved: {save_path}")

    logging.info(f"End traing >>>>>>>> best_val_accuracy: {best_val_accuracy}")

if __name__ == "__main__":
    main()