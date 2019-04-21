import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datagen import DataGenerator

from yolo_loss_new import YoloLossNew
from models import Yolov1_vgg16bn
from warmup_scheduler import GradualWarmupScheduler

from constant import *
from predict import execution

use_gpu = torch.cuda.is_available()

img_folder = 'hw2_train_val/train15000/'
validate_folder = 'hw2_train_val/val1500/'
loss_name = 'test_loss.h5'
model_name = 'test_model.pth'

try:
    os.mkdir('results')
except:
    pass

results_folder = 'results/{}'.format(SAVE_FOLDER)
if os.path.isdir(results_folder) != True:
    os.mkdir(results_folder)

model_path = os.path.join(results_folder, 'models')
if os.path.isdir(model_path) != True:
    os.mkdir(model_path)

stages = dict([(10, 0.0015), (20, 0.001), (30, 0.0005), (40, 0.0001), (50, 0.00001)])
train_num = TRAIN_DATA_SIZE
test_num = VALI_DATA_SIZE
img_size = 448
num_epochs = EPOCH_NUM
lambda_coord = 5
lambda_noobj = .5
n_batch = 24
S = 7
B = 2
C = 16



def save_model_by_epoch(epoch, model):
    if epoch in stages.keys():
        print("Stage {} model saved".format(epoch))
        model_name = 'stage_{}_model.pth'.format(epoch)
        save_torch_model(model, model_name, epoch)

def save_torch_model(model, model_name, epoch):
    path = os.path.join(model_path, model_name)
    torch.save(model.state_dict(), path)
    ## ==========================================
    #   mAP whenever model is saved
    ## ==========================================
    map1 = execution(validate_folder, './Test_hbb', path)
    
    with open(os.path.join(results_folder, 'event_log'), 'a+') as f:
        event_str = 'Model saved at epoch: {}, with mAP: {} \n'.format(epoch, map1)
        f.write(event_str)
    print(event_str)

def main():
    best_test_loss = np.inf
    model = Yolov1_vgg16bn(pretrained = True)
    print('pre-trained vgg16 model has loaded!')

    previous_model_path = model_name
    exists = os.path.isfile(previous_model_path)
    if exists:
        print("Starting from previous result...")
        model.load_state_dict(torch.load(previous_model_path))
    else:
        print("Starting with new train")

    #print(model)

    print('')

    if use_gpu:
        model.cuda()
    
    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) parent_dir, img_size, S, B, C, transforms, num = 15000):

    train_dataset = DataGenerator(
        parent_dir=img_folder, img_size=img_size, 
        S=S, B=B, C=C, 
        transform=transform, num = train_num, train = True)
    
    train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, num_workers=8)
    
    test_dataset = DataGenerator(
        parent_dir=validate_folder,img_size=img_size,
        S=S, B=B, C=C, 
        transform=transform, num = test_num, train = False)
    test_loader = DataLoader(test_dataset,batch_size=n_batch ,shuffle=False,num_workers=8)

    model.train()
    
    train_val_loss_log = open(os.path.join(results_folder, 'train_val_loss_log'), 'w+')
    #loss_fn = YoloLoss(B, S, lambda_coord, lambda_noobj)
    loss_fn = YoloLossNew(B, S, C, lambda_coord, lambda_noobj)
    
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0001, momentum=0.9, weight_decay = 0.0005)
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=30)
    
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        print(epoch, optimizer.param_groups[0]['lr'])
        for i,(img_name, images,target) in enumerate(train_loader):
            #images = images.float()
            #target = target.float()
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()

            optimizer.zero_grad()
            
            pred = model(images)
            loss = loss_fn(pred,target)
            current_loss = loss.item()

            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print("\r%d/%d batches in %d/%d iteration, current error is %f" % (i, len(train_loader), epoch+1, num_epochs, current_loss))
        
        save_model_by_epoch(epoch, model)
        
        # validat on validation set
        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i,(img_name, images,target) in enumerate(test_loader):
                #image = images.float()
                #target = target.float()
                images = Variable(images)
                target = Variable(target)
                if use_gpu:
                    images,target = images.cuda(),target.cuda()
        
                pred = model(images)
                loss = loss_fn(pred,target)
                validation_loss += loss.item()
        
        validation_loss /= len(test_loader)
        # log the training loss and validation loss every epoch
        log_str = 'epoch: {}, train_loss: {}, val_loss: {} \n'.format(epoch+1, current_loss, validation_loss)
        print(log_str)
        train_val_loss_log.writelines(log_str)
        train_val_loss_log.flush()
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            save_torch_model(model, 'best.pth', epoch)

    train_val_loss_log.close()
    event_log.close()

if __name__ == '__main__':
    main()