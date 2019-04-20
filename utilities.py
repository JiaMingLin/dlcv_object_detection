import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import pathlib

from sklearn.preprocessing import OneHotEncoder
from skimage import io
from torch.utils.data import DataLoader
from scipy.misc import imread, imresize
from collections import OrderedDict

from constant import *

def encode():
    pass

def decode_boxes(enc_boxes, img_size = TRAIN_IMAGE_SIZE, mask = None):
    """
    enc_boxes.size() = 7x7x5 [conf, cx, cy, w, h]
    return [conf, xmin, ymin, xmax, ymax]
    """
    if mask is None:
        mask = torch.ByteTensor(GRID_NUM, GRID_NUM)
        mask = ~mask.zero_()
    
    # construct grid index
    grid_length = img_size / float(GRID_NUM)
    g = range(GRID_NUM)
    g_idx = torch.Tensor(np.transpose([np.repeat(g, len(g)), np.tile(g, len(g))])).view(GRID_NUM,GRID_NUM,2)

    # decode box
    dec_boxes = torch.Tensor(enc_boxes.size())
    dec_boxes_center = (enc_boxes[:,:,1:3] + g_idx) * grid_length
    dec_boxes_wh = enc_boxes[:,:, 3:5] * img_size
    dec_boxes[:,:,0] = enc_boxes[:,:,0]
    
    dec_boxes[:,:, 1:3] = dec_boxes_center - 0.5 * dec_boxes_wh
    dec_boxes[:,:, 3:5] = dec_boxes_center + 0.5 * dec_boxes_wh
    
    return dec_boxes[mask]

def label_decoder(enc_label):
    # given one-hot vector
    # return label text
    idx = np.argmax(enc_label)
    for name, code in LABELS.items():
        if code == idx:
            return name
    return 'NoName'

def label_decoder_batch(batch_enc_labels):
    """
        input: batch_enc_labels: (tensor) sized [M, 16]
    """
    text_label_ls = []
    #one_hot = (batch_enc_labels == batch_enc_labels.max())
    for enc_label in batch_enc_labels:
        enc_label = np.array(enc_label)
        if sum(enc_label) != 0:
            text_label_ls.append(label_decoder(enc_label))
    return text_label_ls

def write_bbox_to_file(file_name, bboxes, labels):
    """
        bboxes (tensor): sized [N, 8]
        labels (list): sized N
        file_name = image_file_name
    """
    pass

def main():
    
    
    pass

if __name__ == '__main__':
    main()