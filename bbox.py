#from utilities import *
from datagen import DataGenerator
import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from constant import *

## ===============================================
#  Prediction related functions
## ===============================================

def pred_bbox_revert(pred_bbox_cxcy):
    """
        input:
            1. pred_bbox_cxcy (tensor): sized [98, 4] [cx, cy, w, h]
        output:
            1. pred_bbox_xy (tensor): sized [98, 4] [xmin, ymin, xmax, ymax]
    """
    g = range(GRID_NUM)
    g_idx = torch.Tensor(np.transpose([np.repeat(g, len(g)), np.tile(g, len(g))]))   # sized [49 , 2]
    g_idx = torch.cat((g_idx, g_idx), 1).view(-1,2)   # sized [98 , 2]

    # rescale bbox point back to original size
    grid_length = ORIGINAL_IMAGE_SIZE / float(GRID_NUM)
    #grid_length = TRAIN_IMAGE_SIZE / float(GRID_NUM)
    anchor_points = g_idx * grid_length # sized [98, 2]

    pred_bbox_xy = torch.Tensor(pred_bbox_cxcy.size())
    # xmin, ymin = ((cx, cy) + (ax, ay))*grid_length - (w,h) * original_size
    pred_bbox_xy[:, :2] = (pred_bbox_cxcy[:, :2] * grid_length + anchor_points)  - pred_bbox_cxcy[:, 2:] * ORIGINAL_IMAGE_SIZE
    # xmax, ymax = ((cx, cy) + (ax, ay))*grid_length + (w,h) * original_size
    pred_bbox_xy[:, 2:4] = (pred_bbox_cxcy[:, :2] * grid_length + anchor_points) + pred_bbox_cxcy[:, 2:] * ORIGINAL_IMAGE_SIZE

    return pred_bbox_xy

def bbox_filtering(bbox_xy, bbox_cls_conf, bbox_cls_code, nms_thresh = 0.5, hconf_thresh = 0.1):
    """
        inputs:
            1. bbox_xy (tensor): sized [M, 4], [xmin, ymin, xmax, ymax]
            2. bbox_cls_conf (tensor): sized [M, 1]
            3. bbox_cls_code (tensor): sized [M, 1]
        outputs:
            1. bbox_xy (tensor): sized [N, 4], [xmin, ymin, xmax, ymax]
            2. bbox_cls_conf (tensor): sized [N, 1]
            3. bbox_cls_code (tensor): sized [N, 1]
    """
    ## =============================
    #  filtering by confidence
    ## =============================
    hconf_pass = (bbox_cls_conf > hconf_thresh)  # sized [98, 1] 0 or 1
    # high confidence bbox
    hconf_bbox_xy_pass = hconf_pass.expand_as(bbox_xy)  # sized [98 , 4]
    hconf_bbox_xy = bbox_xy[hconf_bbox_xy_pass].view(-1, 4)  # sized [M, 4]

    # high confidence class confidence
    hconf_cls_conf = bbox_cls_conf[hconf_pass]  # sized [, M]

    # high confidence class prediction
    hconf_max_cls_code = bbox_cls_code[hconf_pass].view(-1,1)   # sized [M, 1]

    ## =============================
    #  NMS filtering
    ## =============================
    nms_pass = nms(hconf_bbox_xy, hconf_cls_conf, threshold = nms_thresh)  # sized [, N]

    nms_pass = nms_pass.view(-1,1)  # sized [N ,1]

    # final bbox
    bbox_xy_final = hconf_bbox_xy[nms_pass].squeeze(1)

    # final class confidence
    cls_conf_final = hconf_cls_conf[nms_pass].squeeze(1)

    # final class probability
    pred_cls_final = hconf_max_cls_code[nms_pass].squeeze(1)
    
    #print(bbox_xy_final.size(), cls_conf_final.size(), pred_cls_final.size())
    
    return bbox_xy_final, cls_conf_final, pred_cls_final
    

def nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4]. x1y1x2y2
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)    

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(int(order))
            break
            
        i = order[0]
        keep.append(i)
        
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)



## ===============================================
# Target related functions
## ===============================================

def target_to_boxes_xy(target):
    """
    input: 
        target(tensor): sized [7,7,26] [cx, cy, w, h, conf, cx, cy, w, h, conf, class * 16]
    
    output:
        1. boxes_xy(list): sized [N, 4], [xmin, ymin, xmax, ymax]
        2. cls_names(list): sized [N, 1], [cls_name]
    """
    if target.size()[0] == 1:
        target = target.squeeze(0)
    
    target = target.view(-1,26)
    
    # mask for containing object grid
    conobj_mask = (target[:,4] > 0)
    conobj_grid = target[conobj_mask.unsqueeze(-1).expand_as(target)].view(-1,26)
    
    # construct grid
    g_idx = grid_construct()  # sized [49, 2]
    conobj_g_idx = g_idx[conobj_mask.unsqueeze(-1).expand_as(g_idx)].view(-1,2)
    
    # rescaling
    grid_length = TRAIN_IMAGE_SIZE / GRID_NUM
    conobj_boxes = conobj_grid[:, 0:4]
    conobj_boxes_center = (conobj_boxes[:, 0:2] + conobj_g_idx) * grid_length
    conobj_boxes_wh = conobj_boxes[:, 2:4] * TRAIN_IMAGE_SIZE
    conobj_boxes_xy = torch.Tensor(conobj_boxes.size()[0], 4)
    conobj_boxes_xy[:, 0:2] = conobj_boxes_center - 0.5 * conobj_boxes_wh   # min
    conobj_boxes_xy[:, 2:4] = conobj_boxes_center + 0.5 * conobj_boxes_wh   # max
    
    # retrieve classe names
    conobj_class = conobj_grid[:, 10:]
    cls_name_ls = label_decoder_batch(conobj_class)
    
    return conobj_boxes_xy.tolist(), cls_name_ls
    
    
def grid_construct():
    
    """
        return g_idx: (tensor) sized [grid_num ^2, 2]
    """
    # grid construct
    grid_num = GRID_NUM
    grid_length = TRAIN_IMAGE_SIZE / GRID_NUM
    g = range(grid_num)
    g_idx = torch.Tensor(np.transpose([np.repeat(g, len(g)), np.tile(g, len(g))]))
    #g_idx = torch.cat((g_idx, g_idx), 1).view(-1,2)
    
    return g_idx

def imshow_with_boxes(img, boxes_xy_ls, cls_ls, conf_ls = None):
    fig,ax = plt.subplots(1)
    if conf_ls is None:
        conf_ls = range(len(boxes_xy_ls))
    
    ax.imshow(img)
    for (box, cls, conf) in zip(boxes_xy_ls, cls_ls, conf_ls):
        xmin = box[0]; ymin = box[1]; xmax = box[2]; ymax = box[3] 
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,edgecolor='r',facecolor='none', label = cls)
        ax.add_patch(rect)

## ========================================================
#  General usage
## ========================================================

def visualize_bbox(image_name, boxes_xy, cls_names, probs, img_size = 448, train = True):
    """
        input: 
            1. image_name(str): input image name, "only name"
            2. boxes_xy(list): [xmin, ymin, xmax, ymax]
            3. cls_names(list): [name]
            4. probs(list):
    """
    image = cv2.imread('hw2_train_val/train15000/images/{}.jpg'.format(image_name))
    if train is False:
        image = cv2.imread('hw2_train_val/val1500/images/{}.jpg'.format(image_name))
    
    image = cv2.resize(image, (img_size, img_size))
    #print((image.numpy()*255).astype(int))
    #result = parse_det(detfile)
    for box_xy, class_name, prob in zip(boxes_xy, cls_names, probs):
        left_up = (int(box_xy[0]), int(box_xy[1]))
        right_bottom = (int(box_xy[2]), int(box_xy[3]))
        color = Color[DOTA_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up, right_bottom ,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    plt.imshow(image)
    cv2.imwrite('result_train.jpg',image)

    
def parse_det(detfile):
    result = []
    with open(detfile, 'r') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 10:
                continue
            x1 = int(float(token[0]))
            y1 = int(float(token[1]))
            x2 = int(float(token[4]))
            y2 = int(float(token[5]))
            cls = token[8]
            prob = float(token[9])
            result.append([(x1,y1),(x2,y2),cls,prob])
    return result 

def visualize_bbox_gt(image_name, train = True, img_size = 448):
    image = cv2.imread('hw2_train_val/train15000/images/{}.jpg'.format(image_name))
    detfile = 'hw2_train_val/train15000/labelTxt_hbb/{}.txt'.format(image_name)
    
    if train is False:
        image = cv2.imread('hw2_train_val/val1500/images/{}.jpg'.format(image_name))
        detfile = 'hw2_train_val/val1500/labelTxt_hbb/{}.txt'.format(image_name)
    
    result = parse_det(detfile)
    image = cv2.resize(image, (img_size, img_size))
    
    for left_up,right_bottom,class_name,prob in result:
        color = Color[DOTA_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
    plt.imshow(image)
    cv2.imwrite('result_gt.jpg',image)
    

def label_decoder(enc_label):
    # given one-hot vector
    # return label text
    idx = np.argmax(enc_label)
    return DOTA_CLASSES[idx]

def label_decoder_batch(batch_enc_labels):
    text_label_ls = []
    for enc_label in batch_enc_labels:
        if enc_label.sum() != 0:
            text_label_ls.append(label_decoder(enc_label))
    return text_label_ls

if __name__ == '__main__':
    
    visualize_bbox_gt('03820')