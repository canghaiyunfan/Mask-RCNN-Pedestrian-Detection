import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# RCNN Dependencies
import utils
import coco
import utils
import model as modellib
import visualize
# Video Detection Dependenices
import cv2
import time
import urllib.request as urllib2
import INRIA_evaluate
import globalvar as gl

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR,"results_output")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def getAllFiles(dirName, houzhui):
    results = []

    for file in os.listdir(dirName):
        file_path = os.path.join(dirName, file)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == houzhui:
            results.append([file_path,os.path.splitext(file)[0]])

    return results

def load_image(imagePath):
    """Load the specified image and return a [H,W,3] Numpy array.
            """
    # Load image
    from PIL import Image
    from PIL import ImageFile
    import imghdr

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if imghdr.what(imagePath) == "png":
        Image.open(imagePath).convert("RGB").save(imagePath)
    image = cv2.imread(imagePath)

    return image

t_prediction = 0
t_start = time.time()
imageNum = 0
imageNames = getAllFiles("/data/data_67L46I1z/Mask-RCNN-Pedestrian-Detection/INRIAPerson/Train/pos", '.png')
person_info = INRIA_evaluate.get_person_num_info("/data/data_67L46I1z/Mask-RCNN-Pedestrian-Detection/INRIAPerson/Train/annotations")
all_person_num = 0
temp = 0
for imageName in imageNames:
    frame =load_image(imageName[0])
    #frame = cv2.imread('test/1.jpg')
    ##################################################
    # Mask R-CNN Detection
    ##################################################
    t = time.time()
    results = model.detect([frame], verbose=1)
    r = results[0]
    t_prediction += (time.time() - t)
    imageNum += 1
    temp += person_info[imageName[1]]
    print('##########################')
    print("imageName:{}.imageNum:{}.".format(imageName[1],imageNum))
    ##################################################
    # Image Plotting
    ##################################################
    visualize.display_instances(frame, r['rois'], r['masks'],
                                r['class_ids'], class_names,
                                r['scores'], title=imageName[1],
                                filepath=OUTPUT_DIR)
    INRIA_evaluate.get_precision(imageName[1], r['class_ids'], person_info[imageName[1]])

print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / imageNum))
print("Total time: ", time.time() - t_start)
print("ground truth行人的总数量:{}".format(gl.get_value('total_person_num_gt')))
print("检测到行人的总数量:{}".format(gl.get_value('total_person_num_detect')))
print(temp)
