import torch
import os
import cv2
import supervision as sv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import pdb 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# IMAGE_NAME = "dog.jpeg"
# IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)

# IMAGE_PATH = '/home/zekunl/Documents/critical-maas/data/cloudfront/19_1.tif'
IMAGE_PATH = '/home/zekunl/Downloads/0004002.jpg'

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)



sam = sam_model_registry[MODEL_TYPE](checkpoint="support_data/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
sam_result = mask_generator.generate(image_rgb)

pdb.set_trace()

mask_annotator = sv.MaskAnnotator()

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

pdb.set_trace()

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)