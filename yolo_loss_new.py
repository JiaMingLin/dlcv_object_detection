import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utilities import *

class YoloLossNew(nn.Module):
    def __init__(self,B = 2, S = 7, C = 16, l_coord = 5, l_noobj = 0.5):
        super(YoloLossNew,self).__init__()
        self.B = B
        self.S = S
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        
    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is 
        [cx,cy,w,h].
        Args:
          box1: (tensor) bounding boxes, sized [2,4].
          box2: (tensor) bounding boxes, sized [2,4].
        Return:
          (tensor) iou, sized [1,1].
        '''
        
        grid_len = TRAIN_IMAGE_SIZE / GRID_NUM
        box1_min_xy = (box1[:, 0:2] * grid_len) - 0.5 * TRAIN_IMAGE_SIZE * box1[:, 2:4] # sized [2,2]
        box2_min_xy = (box2[:, 0:2] * grid_len) - 0.5 * TRAIN_IMAGE_SIZE * box2[:, 2:4] # sized [2,2]        
        
        box1_max_xy = (box1[:, 0:2] * grid_len) + 0.5 * TRAIN_IMAGE_SIZE * box1[:, 2:4] # sized [2,2]
        box2_max_xy = (box2[:, 0:2] * grid_len) + 0.5 * TRAIN_IMAGE_SIZE * box2[:, 2:4] # sized [2,2]
     
        inter_min_xy = torch.max(box1_min_xy, box2_min_xy) # sized [2,2]
        inter_max_xy = torch.min(box1_max_xy, box2_max_xy) # sized [2,2]

        box1_area = self.compute_area(box1_min_xy, box1_max_xy)
        box2_area = self.compute_area(box2_min_xy, box2_max_xy)
        inter_area = self.compute_area(inter_min_xy, inter_max_xy)
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou
    
    def compute_area(self, box_min_xy, box_max_xy):
        w = box_max_xy[:,0] - box_min_xy[:,0] # sized [2,1]
        h = box_max_xy[:,1] - box_min_xy[:,1] # sized [2,1]
    
        pw = torch.zeros(w.size()).cuda(); ph = torch.zeros(h.size()).cuda()
        pw[w > 0] = w[(w > 0)]; ph[h > 0] = h[(h > 0)]
        area = pw * ph
        return area
    
    
    def show_conf(self, pred_tensor):
        pred_tensor2 = pred_tensor.clone().cpu().detach()
        conf_1 = pred_tensor2[:,:,:,0]
        conf_2 = pred_tensor2[:,:,:,5]
        
        area1 = np.mean(np.dot(pred_tensor2[:,:,:,3].numpy(), pred_tensor2[:,:,:,4].numpy()))
        area2 = np.mean(np.dot(pred_tensor2[:,:,:,8].numpy(), pred_tensor2[:,:,:,9].numpy()))
        print(float(conf_1.mean()), float(conf_2.mean()), area1, area2)
    
    def forward(self, pred_tensor, target_tensor):
        """
            input: 
                1. pred_tensor (tensor) sized: [num_batch, S, S , 26], [cx, cy, w, h, conf, cx, cy, w, h, conf, class*16]
                2. target_tensor (tensor) sized: [num_batch, S, S, 26]
            output:
                loss
        """
        #self.show_conf(pred_tensor)
        
        len_box = 5
        len_vec = self.B * len_box + self.C
        num_batch = target_tensor.size()[0]
        
        # object containing mask
        noobj_mask = target_tensor[:,:,:,4] == 0
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # sized [num_batch, S, S , len_vec]
        
        conobj_mask = target_tensor[:,:,:,4] > 0
        conobj_mask = conobj_mask.unsqueeze(-1).expand_as(target_tensor) # sized [num_batch, S, S , len_vec]
        
        # not containing object vectors
        noobj_pred = pred_tensor[noobj_mask].view(-1, len_vec)  # sized [M, len_vec]
        noobj_target = target_tensor[noobj_mask].view(-1, len_vec)  # sized [M, len_vec]
        
        # containing object vectors
        conobj_pred = pred_tensor[conobj_mask].view(-1, len_vec)  # sized [M, len_vec]
        conobj_target = target_tensor[conobj_mask].view(-1, len_vec)  # sized [M, len_vec]
        
        # the responsible box mask
        conobj_pred_boxes = conobj_pred[:, :10].contiguous().view(-1,5)  # sized [2M, 5]
        conobj_target_boxes = conobj_target[:, :10].contiguous().view(-1,5)  # sized [2M, 5]
        
        resp_box_mask = torch.cuda.ByteTensor(conobj_target_boxes.size()).zero_() # the mask must have same size [2M, 5]
        no_resp_box_mask = torch.cuda.ByteTensor(conobj_target_boxes.size()).zero_() # the mask must have same size [2M, 5]
        
        resp_box_iou = torch.zeros(conobj_target_boxes.size()[0]).cuda() # sized [2M, 1]
        for i in range(0, conobj_target_boxes.size()[0], 2):
            # two predict boxes
            box_pred = conobj_pred_boxes[i:i+2].cuda()
            # one gt box
            box_gt = conobj_target_boxes[i:i+2].cuda()
            
            iou = self.compute_iou(box_pred, box_gt)
            max_iou, max_idx = iou.max(0) # max_idx = 0 or 1
            
            resp_box_mask[ i+max_idx, :] = 1
            no_resp_box_mask[i+max_idx-1, :] = 1
            resp_box_iou[i+max_idx] = float(max_iou)   # sized [2M, 1]
        
        resp_pred_boxes = conobj_pred_boxes[resp_box_mask].view(-1, 5) # sized [N, 5]
        resp_target_boxes = conobj_target_boxes[resp_box_mask].view(-1, 5)  # sized [N, 5]

        # responsible boxes' location loss
        resp_pred_boxes_xy = resp_pred_boxes[:, 0:2]  # sized [N, 2]
        resp_pred_boxes_wh = resp_pred_boxes[:, 2:4]  # sized [N, 2]
        resp_target_boxes_xy = resp_target_boxes[:, 0:2]  # sized [N, 2]
        resp_target_boxes_wh = resp_target_boxes[:, 2:4]  # sized [N, 2]
        loc_loss = F.mse_loss(resp_pred_boxes_xy, resp_target_boxes_xy, reduction = 'sum')
        loc_loss += F.mse_loss(torch.sqrt(resp_pred_boxes_wh), torch.sqrt(resp_target_boxes_wh), reduction = 'sum')
        
        # confidence loss
        resp_pred_boxes_conf = resp_pred_boxes[:, 4]    # sized [N, 1]
        # resp_box_iou sized [2M, 1], while resp_box_mask sized [2M, 5]
        resp_box_iou = resp_box_iou.view(-1,1).expand_as(resp_box_mask)    # sized [2M, 5]
        resp_box_iou = resp_box_iou[resp_box_mask].view(-1,5)   # sized [N, 5]
        conf_loss = F.mse_loss(resp_pred_boxes_conf, resp_box_iou[:, 0], reduction = 'sum')
        
        # contain object but not responsible loss
        no_resp_pred_boxes_conf = conobj_pred_boxes[no_resp_box_mask].view(-1,5)   # sized [K,5]
        no_resp_target_boxes_conf = conobj_target_boxes[no_resp_box_mask].view(-1,5)  # sized [K,5]
        no_resp_target_boxes_conf[:, 4] = 0
        conobj_no_resp_loss = F.mse_loss(no_resp_pred_boxes_conf[:,4], no_resp_target_boxes_conf[:,4], reduction = 'sum')
        
        # no object boxes confidence loss
        noobj_pred_conf = noobj_pred[:,4]  # sized [N, 1]
        noobj_target_conf = noobj_target[:,4]   # sized [N, 1]
        noobj_loss = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction = 'sum')
        
        # class loss
        conobj_pred_cls = conobj_pred[:, 10:]  # sized [M, 16]
        conobj_target_cls = conobj_target[:, 10:]  # sized [M, 16]
        cls_loss = F.mse_loss(conobj_pred_cls, conobj_target_cls, reduction = 'sum')
        
        #print("conf_loss: {}, conobj_no_resp_loss: {}".format(conf_loss, conobj_no_resp_loss))
        return (self.l_coord*loc_loss + conf_loss + conobj_no_resp_loss + self.l_noobj*noobj_loss + cls_loss) / num_batch


def test1_cls_loss():
    num_batch = 3
    S = 2; B = 2; C = 3;
    loss = YoloLossNew(B, S, C, 5, 0.5)
    
    # vector length = 13
    x = torch.Tensor([
        [
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,1.3,1.3,1.3,1.3,0,1.3,1.3,1.3,1.3,1.3,1.3,1.3]
        ],
        [
            [0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1],
            [1,1.2,1.2,1.2,1.2,0,1.2,1.2,1.2,1.2,1.2,1.2,1.2]
        ]
    ])
    y = torch.Tensor([
        [
            [1,2,2,2,2,2,2,2,2,2,2,2,2],
            [0,2.1,2.1,2.1,2.1,0,2.1,2.1,2.1,2.1,2.1,2.1,2.1]
        ],
        [
            [0,2.2,2.2,2.2,2.2,2.2,2.2,2.2,2.2,2.2,2.2,2.2,2.2],
            [1,2.2,2.2,2.2,2.2,1,2.2,2.2,2.2,2.2,2.2,2.2,2.2]
        ]
    ])
    z = torch.Tensor([
        [
            [0,3.3,3.3,3.3,3.3,0,3.3,3.3,3.3,3.3,3.3,3.3,3.3],
            [1,3.3,3.3,3.3,3.3,1,3.3,3.3,3.3,3.3,3.3,3.3,3.3]
        ],
        [
            [0,3.1,3.1,3.1,3.1,0,3.1,3.1,3.1,3.1,3.1,3.1,3.1],
            [1,3.1,3.1,3.1,3.1,1,3.1,3.1,3.1,3.1,3.1,3.1,3.1]
        ]
    ])

    w = torch.cat( (x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)), 0)
    
    loss.forward(w,w)
    
def test2_random_vector():
    batch_size = 1
    S = 7
    B = 2
    C = 16
    len_vec = B * 5 + C
    
    # change to the loss class name
    loss_fn = YoloLossNew(B, S, C, 5, 0.5)
    
    loss = 0
    for i in range(3000):
        s1 = torch.Tensor(np.random.laplace(1, 0.05, batch_size * S * S * len_vec)).contiguous().view(batch_size, S, S, len_vec).cuda()
        s2 = torch.Tensor(np.random.laplace(1, 0.05, batch_size * S * S * len_vec)).contiguous().view(batch_size, S, S, len_vec).cuda()
        loss += loss_fn(s1, s2)
        print("round: {}, avg: {}".format(i, loss / i+1))
        
    print("final result: {}".format(loss/3000.))
    

if __name__ == '__main__':
    test2_random_vector()
    