import os
import json
import random

import torch
import os.path as osp
from PIL import Image,ImageFilter
import numpy as np

from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torch.nn.functional as F
import copy

from get_fpn_data import fpn_data

class PanopticNarrativeGroundingValDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg,split,train=False,seed=0,sup_percent=1):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train # True or False
        # split = 'val2017'
        self.split = split # train2017 or val2017

        self.mask_transform = Resize((256, 256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_{:s}.json".format(split))
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}

        # self.panoptic_narrative_grounding = self.load_json(
        #         osp.join(self.ann_dir, 
        #             "png_coco_train2017_unlabeled_dataloader_seed"+str(seed)+'_sup'+str(sup_percent)+'.json')
        # )

        # if not osp.exists(
        #     osp.join(self.ann_dir, 
        #         "png_coco_{:s}_dataloader.json".format(split),)
        # ):
        #     print("No such a dataset")
        # else:
        #     self.panoptic_narrative_grounding = self.load_json(
        #         osp.join(self.ann_dir, 
        #             "png_coco_{:s}_dataloader.json".format(split),)
        #     )
        self.panoptic_narrative_grounding = self.load_json('./ppmn_narr_list.json')
        self.panoptic_narrative_grounding = [
            ln
            for ln in self.panoptic_narrative_grounding
            if (
                torch.tensor([item for sublist in ln["labels"] 
                    for item in sublist])
                != -2
            ).any()
        ]
        fpn_dataset, self.fpn_mapper = fpn_data(cfg, split[:-4])
        self.fpn_dataset = {i['image_id']: i for i in fpn_dataset}

    ## General helper functions
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
    
    def resize_gt(self, img, interp, new_w, new_h):
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((new_w, new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def __len__(self):
        return len(self.panoptic_narrative_grounding)
    
    def vis_item(self, img, gt, idx):
        save_dir = f'vis/{idx}'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        import cv2
        cv2.imwrite(osp.join(save_dir,'img.png'), img.numpy().transpose(1, 2, 0))
        for i in range(len(gt)):
            if gt[i].sum() != 0:
                cv2.imwrite(osp.join(save_dir, f'gt_{i}.png'), gt[i].numpy()*255)
        
    def __getitem__(self, idx):
        localized_narrative = self.panoptic_narrative_grounding[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  
        image_info = self.images[image_id]
        labels = localized_narrative['labels']
        tag_id = int(localized_narrative['tag_id'])

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(noun_vector_padding).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        cur_phrase_index = ret_noun_vector[ret_noun_vector!=0]
        
        _, cur_index_counts = torch.unique_consecutive(cur_phrase_index, return_counts=True)
        cur_phrase_interval = torch.cumsum(cur_index_counts, dim=0)
        cur_phrase_interval = torch.cat([cur_phrase_interval.new_zeros((1)), cur_phrase_interval])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(l)
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(ann_types).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])

        ann_types_valid = ann_types.new_zeros(self.cfg.max_phrase_num)
        ann_types_valid[:len(cur_phrase_interval)-1] = ann_types[cur_phrase_interval[:-1]]


        ann_categories = torch.zeros([
            self.cfg.max_phrase_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0] 
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )
        grounding_instances = torch.zeros(
            [self.cfg.max_phrase_num, image_info['height'], image_info['width']]
        )
        j = 0
        k = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            flag = False
            for b in bbox:
                if b != [0] * 4:
                    flag = True
            if not flag:
                continue
            
            for b in bbox:
                if b != [0] * 4:
                    flag = True
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    if j in cur_phrase_interval[:-1]:
                        grounding_instances[k, :] += instance
                        ann_categories[k] = 1 if \
                                segment_cat["isthing"] else 2
            if j in cur_phrase_interval[:-1]:
                k = k + 1   
            j = j + 1
        assert k == len(cur_phrase_interval) - 1
        grounding_instances = {'gt': grounding_instances}
        ret_noun_vector = {'inter': cur_phrase_interval}

        return caption, grounding_instances, \
            ann_categories, ann_types_valid, noun_vector_padding, ret_noun_vector, fpn_data,tag_id
