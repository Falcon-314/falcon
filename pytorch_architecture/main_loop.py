#==================================
#モジュールのインポート
#==================================
import copy
import math
import gc
import json
import random
import torch
import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns

from tqdm.notebook import tqdm 
from datetime import datetime

import tensorflow as tf

import torch
from torch import nn
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

#=======================================
#時間形式の変換
#=======================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

#===================================
#train用の関数
#===================================
def train_fn(CFG, train_loader, model, criterion, optimizer, scheduler, device, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = data['x'].to(device), data['y'].to(device)
        batch_size = CFG.batch_size
        y_preds = model(inputs)
        loss = criterion(y_preds, targets)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    return losses.avg

#==================================
#valid用の関数
#==================================
def valid_fn(CFG, valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    
    for step, data in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = data['x'].to(device), data['y'].to(device)
        batch_size = CFG.batch_size
        # compute loss
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds, targets)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions
  
  #===============================
  #学習のループ
  #===============================
  def pytorch_table_train(CFG, train, custommodel, get_optimizer, get_scheduler, get_criterion, dataset_train, dataset_preprocess_train, dataset_postprocess_train, get_score, get_result, device, LOGGER):
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            # ====================================================
            # Dataset Preprocess & Dataloader
            # ====================================================
            x_train, y_train, x_valid, y_valid = dataset_preprocess_train(CFG,train,fold)

            train_dataset = dataset_train(x_train.values,y_train.values)
            valid_dataset = dataset_train(x_valid.values,y_valid.values)
            train_loader = DataLoader(train_dataset, 
                                    batch_size=CFG.batch_size, 
                                    shuffle=True, 
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, 
                                    batch_size=CFG.batch_size, 
                                    shuffle=False, 
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
            
            # ====================================================
            # model
            # ====================================================
            model = custommodel(CFG,x_train,CFG.targets)
            model.to(device)

            # ====================================================
            # optimizer setting
            # ====================================================
            optimizer = get_optimizer(CFG, model)

            # ====================================================
            # scheduler setting
            # ====================================================
            scheduler = get_scheduler(CFG,optimizer)

            # ====================================================
            # loss setting
            # ====================================================
            criterion = get_criterion(CFG)

            # ====================================================
            # Setting
            # ====================================================
            if CFG.score_mode == 'max':
                best_score = 0.
                loss = np.inf
            elif CFG.score_mode == 'min':
                best_score = np.inf
                loss = np.inf

            # ====================================================
            # train one loop
            # ====================================================
            LOGGER.info(f"========== fold: {fold} training ==========")
            for epoch in range(CFG.epochs):
                start_time = time.time()       
                # train
                avg_loss = train_fn(CFG, train_loader, model, criterion, optimizer, scheduler, device, epoch)

                # eval
                avg_val_loss, preds = valid_fn(CFG, valid_loader, model, criterion, device)
                            
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

                # scoring
                score = get_score(y_valid, preds)

                elapsed = time.time() - start_time

                LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                LOGGER.info(f'Epoch {epoch+1} - Score: {score}')

                if CFG.score_mode == 'max':
                    if score > best_score:
                        best_score = score
                        LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                        torch.save({'model': model.state_dict(), 
                                    'preds': preds},
                                    CFG.OUTPUT_DIR + f'{CFG.model_name}' + f'_fold{fold}_best.pth')
                        
                if CFG.score_mode == 'min':
                    if score < best_score:
                        best_score = score
                        LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                        torch.save({'model': model.state_dict(), 
                                    'preds': preds},
                                    CFG.OUTPUT_DIR + f'{CFG.model_name}' + f'_fold{fold}_best.pth')

            # ====================================================
            # make oof result
            # ====================================================
            check_point = torch.load(CFG.OUTPUT_DIR + f'{CFG.model_name}' + f'_fold{fold}_best.pth')
            predictions = check_point['preds']

            _oof_df = dataset_postprocess_train(CFG, predictions, train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            score = get_result(_oof_df)
            LOGGER.info(f"========== fold: {fold} result:{score}==========")

    # ====================================================
    # make CV result
    # ====================================================  
    score = get_result(oof_df)
    LOGGER.info(f"========== CVresult:{score}==========")
    oof_df.to_csv(CFG.OUTPUT_DIR+f'oof_df.csv', index=False)
    return oof_df
  
  #==============================
  #予測のループ
  #==============================
  def pytorch_table_inference(CFG, test, custommodel, dataset_test, dataset_preprocess_test, dataset_postprocess_test, device, LOGGER):
    LOGGER.info(f"========== inference ==========")
    predictions = []
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            LOGGER.info(f"========== fold: {fold} inference ==========")
            # ====================================================
            # Dataset Preprocess & Dataloader
            # ====================================================
            x_test = dataset_preprocess_test(CFG, test)

            test_dataset = dataset_test(x_test.values)
            test_loader = DataLoader(test_dataset,
                                     batch_size=CFG.batch_size, 
                                     shuffle=False, 
                                     num_workers=CFG.num_workers, pin_memory=True)

            # ====================================================
            # model
            # ====================================================
            model = custommodel(CFG, x_test, CFG.targets)
            model.to(device)
            model.load_state_dict(torch.load(CFG.OUTPUT_DIR + f'{CFG.model_name}' + f'_fold{fold}_best.pth')['model'])
            
            # ====================================================
            # Setting
            # ====================================================
            tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
            pred = []
            
            # ====================================================
            # inference one loop
            # ====================================================
            for i, data in tk0:
                inputs = data['x'].to(device)
                #inference
                model.eval()
                with torch.no_grad():
                    y_preds = model(inputs)
                pred.extend(y_preds.to('cpu').numpy())
            
            predictions.append(pred)

    # ====================================================
    # averaging
    # ====================================================
    predictions = np.mean(predictions, axis=0)

    # ====================================================
    # make submission
    # ====================================================        
    submission = dataset_postprocess_test(CFG, predictions, test)
    submission.to_csv(CFG.OUTPUT_DIR+f'submission.csv', index=False)
    return submission
