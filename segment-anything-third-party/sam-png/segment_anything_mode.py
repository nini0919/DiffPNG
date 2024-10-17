import numpy as np
import torch
import torch.distributed as dist
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
import os
import sys
from operator import itemgetter
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def tag_masks(masks):
    H,W = masks[0]['segmentation'].shape
    ret_mask = np.zeros((H,W),dtype=np.int64)
    for idx,m in enumerate(masks):
        ret_mask[m['segmentation']]=idx+1
    return ret_mask

def rle2mask(rle_dict):
    height, width = rle_dict["size"]
    mask = np.zeros(height * width, dtype=np.uint8)

    rle_array = np.array(rle_dict["counts"])
    starts = rle_array[0::2]
    lengths = rle_array[1::2]

    current_position = -1
    for start, length in zip(starts, lengths):
     #   current_position += start
        mask[start-1:start-1 + length] = 1
      #  current_position += length

    mask = mask.reshape((height, width), order='F')
    return mask


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 
    1 - mask, 
    0 - background
    
    Returns run length as string formated
    '''

    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    
    runs[1::2] -= runs[::2]
    seg=[]
    
    for x in runs:
        
        seg.append(int(x))
    size=[]
    for x in img.shape:
         size.append(int(x))
    result=dict()
    result['counts']=seg
    result['size']=size
    return result

def save_masks(save_path,masks):
    out_list = []
    for idx,m in enumerate(masks):
        rle_result=mask2rle(m['segmentation'])
        out_list.append(rle_result)
    with open(save_path,'w') as f:
        json.dump(out_list,f)

png_data = json.load(open('ppmn_narr_list.json'))

sam_checkpoint = "./segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
if not osp.exists('./outputs/sam_db'):
    os.mkdir('./outputs/sam_db')
dist.init_process_group(backend="nccl", init_method='env://', world_size=-1, rank=-1, group_name='')

world_size = dist.get_world_size()
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.cuda()
mask_generator = SamAutomaticMaskGenerator(sam)


for idx,data in tqdm(enumerate(png_data),total=len(png_data)):
    if idx % world_size!=local_rank:
        continue
    tag_id = data['tag_id']
    image_id = data['image_id']
    image_path = osp.join("./datasets/coco/val2017","{:012d}.jpg".format(int(image_id)))

    save_path = "./outputs/sam_db/{0}.json".format(idx)
    if os.path.exists(save_path):
        continue
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    save_masks(save_path,masks)
    # masks = sorted(masks, key=itemgetter('area'), reverse=True)
    # sam_mask = tag_masks(masks)
    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()
    # plt.savefig('/home/jjy/NICE_ydn/LAVT-RIS/utils/vis/tmp.png') 
    # plt.clf()
    # show_mask(mask,plt.gca())
    # plt.savefig('/home/jjy/NICE_ydn/LAVT-RIS/utils/vis/tmp1.png') 
    # import random
    # seg_map = sam_mask
    # num_classes = np.max(seg_map) + 1
    # colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(num_classes)]

    # seg_img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    # for i in range(seg_map.shape[0]):
    #     for j in range(seg_map.shape[1]):
    #         seg_img[i, j] = colors[seg_map[i, j]]

    # plt.imshow(seg_img)
    # plt.axis('off')
    # plt.show()
    # plt.savefig('tmp.png')
    
    # np.save('/home/jjy/NICE_ydn/LAVT-RIS/anns/{0}/sam_masks/{1}.npy'.format(dataset,mask_id),sam_mask.astype(np.float32))
