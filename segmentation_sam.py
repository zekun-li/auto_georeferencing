import torch
import os
import cv2
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from matplotlib import pyplot as plt
import numpy as np
import pdb 

def load_and_resize_image(input_image_path, max_size):
    # Read the image using OpenCV
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get the original width and height
    height, width, _ = img.shape
    
    # Determine the scaling factor to fit the longer side to max_size
    if width >= height:
        scaling_factor = max_size / width
    else:
        scaling_factor = max_size / height
    
    # Calculate the new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return img, resized_img, scaling_factor


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

checkpoint_path = "support_data/sam_vit_h_4b8939.pth"
img_max_size = 200

input_image_path = '/home/zekunl/Downloads/0004002.jpg'


def run_sam(input_image_path):
    image, resized_img, scaling_factor = load_and_resize_image(input_image_path, max_size=img_max_size)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_result = mask_generator.generate(resized_img)


    area_list = []
    index_list = []
    for idx in range(len(sam_result)):
        re = sam_result[idx]
        index_list.append(idx)
        area_list.append(re['area'])

    # sort the segmentation masks by area
    sorted_area_list = [x for x, y in sorted(zip(area_list, index_list))][::-1]
    sorted_index_list = [y for x, y in sorted(zip(area_list, index_list))][::-1]

    diff_area_list = [sorted_area_list[i-1] - sorted_area_list[i] for i in range(1, len(sorted_area_list))]


    # get the segmentation for map plot area
    map_plot_area = sam_result[sorted_index_list[np.argmax(diff_area_list)]]
    seg_mask = map_plot_area['segmentation']
    bbox = map_plot_area['bbox']
