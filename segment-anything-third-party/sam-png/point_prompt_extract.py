# This file aims to extract point prompt from annotation of refcoco dataset

import os
import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from refer import REFER

sys.path.append("..")

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

img_path = "/home/jjy/NICE_ydn/LAVT-RIS/refer/data/images/mscoco/images/train2014"
annotation_path = '/home/jjy/NICE_ydn/LAVT-RIS/anns/refcoco/refcoco_99%_image.json'
sam_checkpoint = "/home/jjy/NICE_ydn/LAVT-RIS/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
dataset='refcoco'
device = "cuda"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

with open(annotation_path) as f:
    ref_data = json.load(f)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
predictor = SamPredictor(sam)

for idx,ref in tqdm(enumerate(ref_data['train']),total=len(ref_data['train'])):
    image_id = ref['iid'] # image_id
    ref_bbox = ref['bbox']
    cat_id = ref['cat_id']
    mask_id = ref['mask_id']

    if os.path.exists('/home/jjy/NICE_ydn/LAVT-RIS/anns/{0}/sam_masks/{1}.npy'.format(dataset,mask_id)):
        continue
    img_path = os.path.join("/home/jjy/NICE_ydn/LAVT-RIS/refer/data/images/mscoco/images/train2014",'COCO_train2014_%012d.jpg'%image_id)
    mask = np.load(os.path.join("/home/jjy/NICE_ydn/LAVT-RIS/anns/{0}/masks/{1}".format(dataset,dataset),'%d.npy'%mask_id))
    # plt.imshow(mask)
    # plt.savefig('/home/jjy/NICE_ydn/LAVT-RIS/utils/vis/sam/test_5_mask_{0}_gt.png'.format(0))  
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 点的选取 ,从mask中随机选取一个点
    y_arr,x_arr = np.where(mask>0)
    rand_y = np.random.randint(0,len(y_arr)-1)
    rand_x = np.random.randint(0,len(x_arr)-1)
    input_point = np.array([[x_arr[rand_x],y_arr[rand_y],]])
    input_label = np.array([1])
    predictor.set_image(image)
    # point
    sam_mask, score, logit = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    np.save('/home/jjy/NICE_ydn/LAVT-RIS/anns/{0}/sam_masks/{1}.npy'.format(dataset,mask_id),sam_mask.astype(np.float32))
    