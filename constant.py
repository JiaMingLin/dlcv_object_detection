import torch

GRID_NUM = 7
TRAIN_IMAGE_SIZE = 448
ORIGINAL_IMAGE_SIZE = 512
CLASS_NUM = 16
NUM_WORKERS = 2

VALI_DATA_SIZE = 1500
TRAIN_DATA_SIZE = 1500
EPOCH_NUM = 40

NMS_THRESH = 0.5
HCONF_THRESH = 0.2

use_gpu = torch.cuda.is_available()

MODELS = {
    'vgg16_test_3': './models/model_test_3/models/best.pth',
    'vgg16_test_4': './models/model_test_4_bug_fix/models/best.pth',
    'vgg16_test_5': './models/model_test_4_bug_fix/models/stage_30_model.pth'
}


DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
