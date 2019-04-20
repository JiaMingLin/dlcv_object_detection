from utilities import *
from transformations import *
from constant import *
import cv2
import sys

training_image_path = 'hw2_train_val/train15000/images/'
training_target_path = 'hw2_train_val/train15000/labelTxt_hbb/'

validation_image_path = 'hw2_train_val/val1500/images/'
validation_target_path = 'hw2_train_val/val1500/labelTxt_hbb/'


class DataGenerator(data.Dataset):
    def __init__(self, parent_dir, img_size, S, B, C, transform, num = 15000, train = True):
        self.parent_dir = parent_dir
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.cnt = 0
        self.train = train
        
        self.image_names = []
        self.imgs = []
        self.targets = [] # xmin, ymin, xmax, ymax, class_num
        self.mean = (123,117,104)#RGB
        
        # loading training data
        image_path = os.path.join(parent_dir, "images")
        files = os.listdir(image_path)
        
        for f in files:
            if self.cnt == num:
                 break;
            name = f.replace(".jpg", "")
            
            # read image
            img = cv2.imread(os.path.join(image_path, f))
            
            # retrieve targets
            targets_per_img = self.read_target(parent_dir, name)
            
            num_bbox = len(targets_per_img)
            # cache name, image and boxes(num_bbox > 0) and   and (num_bbox > 0) and (num_bbox <= 20 )
            if train is True:
                self.image_names.append(name)
                self.imgs.append(img)
                self.targets.append(targets_per_img)
                self.cnt+=1
            elif train is False:
                self.image_names.append(name)
                self.imgs.append(img)
                self.targets.append(targets_per_img)
                self.cnt+=1

            if self.cnt % 500 == 0:
                sys.stdout.write(' {} '.format(self.cnt))
                sys.stdout.flush()

    def read_target(self,parent_dir, target_file_name):
        """
            output: list([xmin, ymin, xmax, ymax, cls_num])
        """
        
        path = os.path.join(parent_dir, "labelTxt_hbb", target_file_name + ".txt")
        target_per_img = []
        with open(path) as fin:
            for line in fin:
                splited = line.replace('\n', '').split()
                #if splited[-1] == '0':
                xmin = float(splited[0])
                ymin = float(splited[1])
                xmax = float(splited[4])
                ymax = float(splited[5])
                cls_num = DOTA_CLASSES.index(splited[8])
                target_per_img.append([xmin, ymin, xmax, ymax, cls_num])                
           
        return torch.Tensor(target_per_img)

    
    def __getitem__(self, index):
        """
            arg: index
            return 
                1. image_name
                2. resized image
                3. 7x7x26 (tensor)
        
        """
        name = self.image_names[index]   # str
        img = self.imgs[index]
        
        img = adaptiveHE(img)
        img = sharpening(img)
        
        # boxes and labels
        raw_target = self.targets[index].clone()  # [xmin, ymin, xmax, ymax, cls_num]
        try:
            boxes = raw_target[:, 0:4]
            labels = raw_target[:, 4]
        except:
            boxes = torch.zeros(1, 4)
            labels = torch.zeros(1)

        img = adaptiveHE(img)        
        #
        # data augmentation
        
        if self.train:
            img, boxes = resize(img, boxes, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
            #img, boxes = random_flip(img, boxes)
            #img, boxes = randomScale(img,boxes)
            #img = randomBlur(img)
            #img = RandomBrightness(img)
            #img = RandomHue(img)
            #img = RandomSaturation(img)
        else:
            img, boxes = resize(img, boxes, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
            #img, boxes = center_crop(img, boxes, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))

        img = BGR2RGB(img)
        #img = subMean(img,self.mean) 
        img = self.transform(img)

        target = torch.zeros((GRID_NUM,GRID_NUM,26))
        cell_size = TRAIN_IMAGE_SIZE/GRID_NUM
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = torch.ceil(cxcy_sample/cell_size)-1 #
            target[int(ij[0]),int(ij[1]),4] = 1
            target[int(ij[0]),int(ij[1]),9] = 1
            target[int(ij[0]),int(ij[1]),int(labels[i])+10] = 1
            xy = ij*cell_size # regard to the left top
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[0]),int(ij[1]),2:4] = wh[i] / TRAIN_IMAGE_SIZE
            target[int(ij[0]),int(ij[1]),0:2] = delta_xy
            target[int(ij[0]),int(ij[1]),7:9] = wh[i] / TRAIN_IMAGE_SIZE
            target[int(ij[0]),int(ij[1]),5:7] = delta_xy 
        return name, img, target
    
    def __len__(self):
        return len(self.targets)

def main():
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    img_folder = 'hw2_train_val/train15000/'
    train_dataset = DataGenerator(parent_dir=img_folder, img_size=448, S=7, B=2, C=16, transform=transform, num = 200)    
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
    
    train_loader.__getitem__(27)
    
    train_iter = iter(train_loader)
    img, target = next(train_iter)
    for i in range(7):
        for j in range(7):
            if target[0,i,j,0] != 0:
                print(i,j)
                print(target[0,i,j])

    img, target = next(train_iter)

    print(img.size())
    print(target.size())
    img, target = next(train_iter)
    print(img.size())
    print(target.size())
    
    
if __name__ == '__main__':
    main()