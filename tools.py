#!/usr/bin/env python
# coding: utf-8

import torch
import shutil
import os

def accuracy(logits, labels):
    pre_labels = logits.argmax(1)
    acc = float((pre_labels==labels).sum())/len(labels)
    return acc

def save_model(model, optimizer, scheduler, global_step, m = 1):
    filename = os.listdir('./model'+str(m))[0]
    shutil.rmtree(os.path.join('./model'+str(m), filename))
    file = os.path.join('./model'+str(m), str(global_step))
    os.mkdir(file)
    torch.save(model.state_dict(),os.path.join(file,'model.pt'))
    torch.save(optimizer.state_dict(),os.path.join(file,'optimizer.pt'))
    torch.save(scheduler.state_dict(),os.path.join(file,'scheduler.pt'))
    print('模型保存成功！')

def load_model(model = None, optimizer = None, scheduler = None, m = 1):
    filename = os.listdir('./model'+str(m))[0]
    model.load_state_dict(torch.load(os.path.join('./model'+str(m),filename)+'/model.pt'))
    if optimizer != None:
        optimizer.load_state_dict(torch.load(os.path.join('./model'+str(m),filename)+'/optimizer.pt'))
    if scheduler != None:
        scheduler.load_state_dict(torch.load(os.path.join('./model'+str(m),filename)+'/scheduler.pt'))
    print('模型参数加载成功！')
