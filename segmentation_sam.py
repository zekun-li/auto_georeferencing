import torch
import os
import cv2
import skimage.transform as st
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from matplotlib import pyplot as plt
import numpy as np
import pdb 



MODEL_TYPE = "vit_h"

checkpoint_path = "support_data/sam_vit_h_4b8939.pth"



def resize_img(img, max_size = 200):
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
    
    return resized_img, scaling_factor


def load_and_resize_image(input_image_path, max_size):
    # Read the image using OpenCV
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    resized_img, scaling_factor = resize_img(img, max_size)

    return img, resized_img, scaling_factor

def get_map_area_by_area_diff(sam_result):
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

    return map_plot_area

def get_map_area_by_iou(sam_result):
    sorted_anns = sorted(sam_result, key=(lambda x: x['predicted_iou']), reverse=True)

    map_plot_area = sorted_anns[0]

    return map_plot_area


def run_sam(image, resized_img, scaling_factor, device):

    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    sam_result = mask_generator.generate(resized_img)

    
    map_plot_area = get_map_area_by_iou(sam_result)
    
    seg_mask = map_plot_area['segmentation']
    bbox = map_plot_area['bbox']
    
    # pdb.set_trace()
    seg_mask = np.array(seg_mask * 255, dtype=np.uint8)
    seg_mask = st.resize(seg_mask, (image.shape[0], image.shape[1]), order=0, preserve_range=True, anti_aliasing=True)
    bbox = [int(a/scaling_factor) for a in bbox]

    return seg_mask, bbox


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_image_path = '/home/zekunl/Downloads/0004002.jpg'

    image, resized_img, scaling_factor = load_and_resize_image(input_image_path, max_size=img_max_size)

    seg_mask, bbox = run_sam(image, resized_img, scaling_factor, device)

    pdb.set_trace()

    print(bbox)

if __name__ == '__main__':
    main()