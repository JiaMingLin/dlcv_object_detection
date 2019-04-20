from torch.autograd import Variable

from utilities import *
from bbox import *
from models import Yolov1_vgg16bn
from constant import *

import shutil

def predict(img, model, DEBUG = False, dummy_example = None):
    """
        input: 
            1. image (tensor) sized: [3, 448, 448]
            2. model
            3. DEBUG (boolean)
            4. dummy_example (tensor) sized [3, 448, 448]
        output:
            1. pred_bbox_cxcy (tensor), sized [98, 4], [cx,cy,w,h]
            2. pred_class_conf (tensor), sized [98, 1], max class prob * IoU confidence
            3. pred_max_cls_code (tensor), sized [98, 16]
    """
    img = img.unsqueeze(0)
    img = Variable(img)
    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        pred = model(img)   # pred sized [1,7,7,26]
    
    pred = pred.squeeze(0).view(-1, 26).cpu()    # sized [49, 26]
    
    if DEBUG is True:
        pred = dummy_example.view(-1,26)

    pred_bboxes = pred[:,:10].contiguous().view(-1, 5)  # sized [98, 5]
    pred_obj_conf = pred_bboxes[:, 4].unsqueeze(1)   # sized [98, 1]
    pred_cls_prob = pred[:, 10:]    # sized [49, 16]
    pred_cls_prob = torch.cat((pred_cls_prob, pred_cls_prob), 1).view(-1, 16)  # sized [98, 16]
    
    # bbox cx, cy, w, h
    pred_bbox_cxcy = pred_bboxes[:, :4]  # sized [98, 4]
    
    # max class probability
    pred_max_cls_prob = torch.max(pred_cls_prob, 1)[0].view(-1,1)    # sized [98, 1]
    pred_max_cls_code = torch.max(pred_cls_prob, 1)[1].view(-1,1)
    # class confidence
    pred_cls_conf = pred_obj_conf.mul(pred_max_cls_prob)  # sized [98, 1] 
    
    return pred_bbox_cxcy, pred_cls_conf, pred_max_cls_code


def predict_all(input_path, model_path, data_size = 1500, num_workers = 2):
    
    ## ==========================
    #  Data
    ## ==========================
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    # validation dataset loader
    validation_dataset = DataGenerator(
        parent_dir = input_path, img_size = TRAIN_IMAGE_SIZE,
        S = GRID_NUM, B=2, C = CLASS_NUM, 
        transform=transform, num = data_size, train = False
    )
    validation_loader = DataLoader(validation_dataset, batch_size = 1, shuffle = False, num_workers = num_workers)
    
    ## ==========================
    #  Model
    ## ==========================
    model = model_inport(model_path)
    
    ## ==========================
    #  Predict All
    ## ==========================
    prediction_results = [] # [image_name, pred_bbox_xy, cls_conf, max_cls_prob]
    data_counts = len(validation_dataset)
    for image_id in range(data_counts):
        img_name, images , target = validation_dataset.__getitem__(image_id)
        print("Detecting objects in image: {}....".format(img_name))
        pred_bbox_cxcy, cls_conf, max_cls_code = predict(images, model)
        pred_bbox_xy = pred_bbox_revert(pred_bbox_cxcy)
        
        prediction_results.append(
            (
                img_name,
                pred_bbox_xy,
                cls_conf,
                max_cls_code
            )
        )
        
    ## ==========================
    #  Filtering
    ## ==========================
    filtered_results = []
    for image_name, pred_bbox_xy, cls_conf, max_cls_code in prediction_results:
        
        bbox_xy_final,cls_conf_final,pred_cls_code_final = bbox_filtering(pred_bbox_xy, cls_conf, max_cls_code
                                                                         ,nms_thresh = NMS_THRESH, hconf_thresh = HCONF_THRESH)
        
        # class names
        #max_cls_idx = torch.max(pred_cls_final, 1)[1].tolist() if pred_cls_final.size(0) != 0 else []
        cls_labels = [DOTA_CLASSES[idx] for idx in pred_cls_code_final.squeeze(1).tolist()]
        
        filtered_results.append(
            (
                image_name, 
                bbox_xy_final.tolist(),
                cls_conf_final.tolist(),
                cls_labels
            )
        )
        
    return filtered_results

def model_inport(model_path):
    #model_path = os.path.join('models', model_name + '.pth')
    #model = MODELS[model_name]
    model = Yolov1_vgg16bn(pretrained = True)
    model.load_state_dict(torch.load(model_path))
    
    if use_gpu: 
        model.cuda()
        
    model.eval()
    return model

def format_out(number):
    
    number = int(number)
    if number < 0 :
        number = 0
    return str(number)

def write_predictions_to_file(predicted_results, output_folder):
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for (image_name, bbox_xy_final, cls_conf_final, cls_labels) in predicted_results:
        print("Writing results for image: {}.....".format(image_name))
        with open('{}/{}.txt'.format(output_folder,image_name), 'w+') as f:
            for box, cls_conf, label in zip(bbox_xy_final, cls_conf_final, cls_labels):
                # box [0 xmin, 1 ymin, 2 xmax, 3 ymax]
                xmin = format_out(box[0]); ymin = format_out(box[1]);
                xmax = format_out(box[2]); ymin = format_out(box[1]);
                xmax = format_out(box[2]); ymax = format_out(box[3]);
                xmin = format_out(box[0]); ymax = format_out(box[3]);
                box_write = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                write_str = ' '.join(box_write)
                write_str += (' ' + label)
                write_str += (' ' + str(cls_conf))
                write_str += '\n'
                f.write(write_str)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_img_folder", help="The input image folder")
    parser.add_argument("output_pred_folder", help="The output prediction folder")
    parser.add_argument("model", help="The name of backbone model")
    args = parser.parse_args()
    
    # inputs
    train_folder = args.input_img_folder
    output_folder = args.output_pred_folder
    model_path = MODELS[args.model]
    
    print("Traing set folder: {}".format(train_folder))
    print("Results output folder: {}".format(output_folder))
    print("Model File: {}".format(model_path))
    
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    
    execution(train_folder, output_folder, model_path)
    

def execution(train_folder, output_folder, model_path):

    print("Object detection starting........")
    predicted_results = predict_all(train_folder, model_path, data_size = VALI_DATA_SIZE, num_workers = NUM_WORKERS)
    
    print("Writeing results into folder {}".format(output_folder))
    write_predictions_to_file(predicted_results, output_folder)
    
    import hw2_evaluation_task2
    
    hw2_evaluation_task2.main()
    
if __name__ == '__main__':
    main()