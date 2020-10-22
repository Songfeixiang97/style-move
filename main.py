#!/usr/bin/env python
# coding: utf-8


from train import Train

t = Train(lr = 0.000001,pretrain = True)
t.train(1000,500,100)
