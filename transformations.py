import math
import random

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import cv2
import numpy as np

def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    hw = img.shape
    w = hw[1]; h = hw[0]
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return cv2.resize(img, (size[0], size[1])), \
           boxes*torch.Tensor([sw,sh,sw,sh])

def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x,y,x,y])
    boxes[:,0::2].clamp_(min=0, max=w-1)
    boxes[:,1::2].clamp_(min=0, max=h-1)
    return img, boxes

def center_crop(img, boxes, size):
    '''Crops the given PIL Image at the center.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).
    Returns:
      img: (PIL.Image) center cropped image.
      boxes: (tensor) center cropped boxes.
    '''
    hw = img.shape
    w = hw[1]; h = hw[0]
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j+ow, i+oh))
    boxes -= torch.Tensor([j,i,j,i])
    boxes[:,0::2].clamp_(min=0, max=ow-1)
    boxes[:,1::2].clamp_(min=0, max=oh-1)
    return img, boxes


def BGR2RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def random_flip(im, boxes):
    im_lr = im
    if random.random() < 0.5:
        im_lr = np.fliplr(im).copy()
        h,w,_ = im.shape
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:,0] = xmin
        boxes[:,2] = xmax
    return im_lr, boxes

def random_bright(  im, delta=16):
    alpha = random.random()
    if alpha > 0.3:
        im = im * alpha + random.randrange(-delta,delta)
        im = im.clip(min=0,max=255).astype(np.uint8)
        return im
    return im

def randomScale( bgr,boxes):
    #固定住高度，以0.8-1.2伸缩宽度，做图像形变
    if random.random() < 0.5:
        scale = random.uniform(0.8,1.2)
        height,width,c = bgr.shape
        bgr = cv2.resize(bgr,(int(width*scale),height))
        scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        return bgr,boxes
    return bgr,boxes

def randomBlur( bgr):
    if random.random()<0.5:
        bgr = cv2.blur(bgr,(5,5))
    return bgr

def RandomHue( bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        h = h*adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = HSV2BGR(hsv)
    return bgr

def RandomSaturation( bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        s = s*adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = HSV2BGR(hsv)
    return bgr

def RandomBrightness( bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        v = v*adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = HSV2BGR(hsv)
    return bgr

def subMean(bgr,mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr

def BGR2HSV( img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def HSV2BGR( img):
    return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

def adaptiveHE(image):
    image_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_to_yuv[:,:,0] = clahe.apply(image_to_yuv[:,:,0])
    image_to_yuv[:,0,:] = clahe.apply(image_to_yuv[:,0,:])
    image_to_yuv[0,:,:] = clahe.apply(image_to_yuv[0,:,:])
    adaptiveHE1 = cv2.cvtColor(image_to_yuv, cv2.COLOR_YUV2BGR)
    return adaptiveHE1

def sharpening(image):
    image_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    image_to_yuv[0,:,:] = cv2.filter2D(image_to_yuv[0,:,:], -1, kernel_sharpening)
    image_to_yuv[:,0,:] = cv2.filter2D(image_to_yuv[:,0,:], -1, kernel_sharpening)
    image_to_yuv[:,:,0] = cv2.filter2D(image_to_yuv[:,:,0], -1, kernel_sharpening)

    sharpened = cv2.cvtColor(image_to_yuv, cv2.COLOR_YUV2BGR)
    return sharpened

def draw(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.show()