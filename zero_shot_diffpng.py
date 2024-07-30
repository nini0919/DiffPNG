import os.path as osp
import gc
import json
import re

from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from scheduler_dev import DDIMSchedulerDev
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from png_dataset import PanopticNarrativeGroundingValDataset
import argparse
from collate_fn import default_collate
from detectron2.structures import ImageList
from meters import average_accuracy
import os
import shutil
import transformers

def compute_mask_IoU(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union, intersection/union

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training and testing pipeline."
    )
    
    # setting
    parser.add_argument(
        '--training', 
        action='store_true', 
        help='Training enable.'
    )
    parser.add_argument(
        '--local_rank', 
        type=int, 
        help='Local rank for ddp.'
    )
    parser.add_argument(
        '--backend', 
        default='nccl', 
        type=str, 
        help='Backend for ddp.'
    )
    parser.add_argument(
        '--seed', 
        default=3407, 
        type=int, 
        help='Random Seed.'
    )
    parser.add_argument(
        '--num_gpus',
        default=4, 
        type=int,
        help='Number of GPUs to use (applies to both training and testing).'
    )

    # model
    parser.add_argument(
        '--detectron2_ckpt', 
        default='./pretrained_models/fpn/model_final_cafdb1.pkl', 
        type=str, 
        help='ckpt path of fpn from detectron2.'
    )
    parser.add_argument(
        '--detectron2_cfg', 
        default='./configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_train.yaml',
        type=str, 
        help='cfg path of fpn from detectron2.'
    )
    parser.add_argument(
        '--max_sequence_length',
        default=230,
        type=int,
        help='Max length of the input language sequence.'
    )
    parser.add_argument(
        '--max_seg_num',
        default=64,
        type=int,
        help='Max num of the noun phrase to be segmented.'
    )
    parser.add_argument(
        '--max_phrase_num',
        default=30,
        type=int,
        help='Max num of the noun phrase to be segmented.'
    )
    # data
    parser.add_argument(
        '--data_path',
        default='./datasets/coco', 
        type=str,
        help='The path to the data directory.'
    )
    
    parser.add_argument(
        '--data_dir',
        default='./datasets', 
        type=str,
        help='The path to the data directory.'
    )

    parser.add_argument( 
        '--output_dir', 
        default="./output", 
        type=str, 
        help='Saving dir.'
    )
    parser.add_argument(
        '--self_enhanced', 
        type=bool,
        default=False,
        help='.'
    )
    parser.add_argument(
        '--sam_enhanced', 
        type=bool,
        default=False,
        help='.'
    )
    parser.add_argument(
        "--self_res",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--cross_res",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--tao",
        type=float,
        default=0.5,
    )
    return parser.parse_args()

def split_text(text):
    words_and_punctuation = re.findall(r"[\w']+|[.,!?;]", text)
    return words_and_punctuation

def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        # for tensor in tensors:
        #     tensor.mul_(1.0 / world_size)
        for i in range(len(tensors)):
            tensors[i] = torch.mul(tensors[i],1.0 / world_size)
    return tensors


def upsample_eval(tensors, pad_value=0, t_size=[400, 400]):
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(t_size)
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union

def find_nearest_period_index(word_list):
    target_index = 74
    nearest_period_index = None

    for i, word in enumerate(word_list[:75]):
        if word == '.':
            nearest_period_index = i
        elif i == target_index:
            break

    return nearest_period_index

def split_sentences(token_list):
    assert len(token_list)>75
    
    splited_sentences = []
    while len(token_list)>75:
        s_end_idx = find_nearest_period_index(token_list)
        if s_end_idx is None:
            splited_sentences.append(token_list[:75])
            token_list = token_list[75:]
        else:
            splited_sentences.append(token_list[:s_end_idx+1])
            token_list = token_list[s_end_idx+1:]
    if len(token_list)!=0:
        splited_sentences.append(token_list)
    return splited_sentences

def aggregate_cross_attention(ldm_stable,tokens,cross_attn,selected_nouns_clip_idx,tag_id,noun_idx):
    if not osp.exists(f'./outputs/scores'):
        os.mkdir(f'./outputs/scores')
    if not osp.exists(f'./outputs/scores/{tag_id}_{noun_idx}.pt'):
        noun_text_embeddings = []
        for n in tokens:
            text_input = ldm_stable.tokenizer([n],padding="max_length", max_length=77,truncation=True,return_tensors='pt')
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))
            noun_text_embeddings.append(text_embeddings[1])
        noun_text_embeddings = torch.concat(noun_text_embeddings)
        scores = noun_text_embeddings@noun_text_embeddings.T
        scores = scores - scores.min()
        scores = scores / scores.max()
        weighted_s = scores[-1].softmax(dim=-1)
        torch.save(weighted_s,f'./outputs/scores/{tag_id}_{noun_idx}.pt')
    else:
        weighted_s = torch.load(f'./outputs/scores/{tag_id}_{noun_idx}.pt')
    weighted_cross_attn = torch.zeros_like(cross_attn[0])
    for i in range(len(selected_nouns_clip_idx)):
        weighted_cross_attn += torch.tensor(weighted_s[i]*cross_attn[selected_nouns_clip_idx[i]])
    return weighted_cross_attn

def dist_evaluate(cfg,data_loader,device):
    scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
    ldm_stable =  StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    tokenizer = ldm_stable.tokenizer
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    cnt = 0
    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []

    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector,fpn_input_data,tag_id)) in tqdm(enumerate(data_loader)):
        ann_categories = ann_categories
        ann_types = ann_types
        image_id = fpn_input_data[0]['image_id']
        tag_id = tag_id.item()
        grad_sample = ann_types != 0
        enc_inp = list(ldm_stable.tokenizer(
                caption[0],
                max_length=230,
                return_tensors="pt",
        ).input_ids.numpy())[0]
        decoder = ldm_stable.tokenizer.decode
        tokens = [ decoder(k) for k in enc_inp[1:-1]]
        tokens_bak = tokens.copy()
        valid_phrase = []  # ['woman','teddy bear toy']
        bert_token_ids = torch.tensor(bert_tokenizer(caption,max_length=230)['input_ids'])
        pad_bert_token_ids = F.pad(bert_token_ids,(0,230-bert_token_ids.shape[1]))
        noun_tokens_ids = pad_bert_token_ids[0,torch.tensor(noun_vector_padding[0]).nonzero()]
        noun_tokens = [bert_tokenizer.decode(n) for n in list(noun_tokens_ids.cpu().numpy())]
        words_list =  [bert_tokenizer.decode(n) for n in list(bert_token_ids.reshape(-1,1).numpy())][1:-1]

        phrase_list = []
        valid_phrase_idx =[]            
        valid_noun_vector = noun_vector_padding[0][1:len(bert_token_ids[0])-1]
        cur_phrase = None
        valid_phrase_bert_token_ids = []
        cur_phrase_bert_token_ids = []

        k = 0 
        tokens_length = len(words_list)
        while k<tokens_length:
            if k<tokens_length-1:
                if cur_phrase is None:
                    cur_phrase = words_list[k]
                else:
                    cur_phrase = cur_phrase + ' '+ words_list[k]
                cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
            elif k==tokens_length-1:
                if valid_noun_vector[k].item()!=valid_noun_vector[k-1].item():
                    phrase_list.append(words_list[-1])
                    valid_phrase_idx.append(valid_noun_vector[k].item())
                    valid_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
                else:
                    cur_phrase = cur_phrase + ' '+ words_list[k]
                    cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())

                    phrase_list.append(cur_phrase)
                    valid_phrase_idx.append(valid_noun_vector[k].item())
                    valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
                    cur_phrase = None
                    cur_phrase_bert_token_ids = []
            if k< tokens_length-1 and valid_noun_vector[k]!=valid_noun_vector[k+1]:
                valid_phrase_idx.append(valid_noun_vector[k].item())
                phrase_list.append(cur_phrase)
                valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
                cur_phrase = None
                cur_phrase_bert_token_ids = []
            k+=1

        cnt+=1
        phrase_list = []
        for k in valid_phrase_bert_token_ids:
            if type(k)!=list:
                k = [k]
            phrase_list.append(bert_tokenizer.decode(k))
        if len(tokens)<=75:
            splited_tokens = [tokens]
        else:
            splited_tokens = split_sentences(tokens)
        cross_attention = load_cross_attention(tokenizer,splited_tokens,tag_id,cfg.cross_res)
        clip_phrase_list_idx = []
        for p in phrase_list:
            if p=="'s":
                clip_phrase_list_idx.append([568])
            elif p=="' s":
                clip_phrase_list_idx.append([568])
            elif "' s" in p:
                clip_phrase_list_idx.append(tokenizer(p.replace("' s","'s"))['input_ids'][1:-1])
            else:
                clip_phrase_list_idx.append(tokenizer(p)['input_ids'][1:-1])
        
        phrase_tokens = []
        selected_nouns_clip_idx = []
        cum = 0
        for i in range(len(clip_phrase_list_idx)):
            tmp=[]
            tokens = []
            if valid_phrase_idx[i]>0:
                for j in range(len(clip_phrase_list_idx[i])):
                    tokens.append(decoder(clip_phrase_list_idx[i][j]))
                    tmp.append(cum)
                    cum+=1
                selected_nouns_clip_idx.append(tmp)
                phrase_tokens.append(tokens)
            else:
                cum+=len(clip_phrase_list_idx[i])

        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
            gts = upsample_eval(gts)
        
        predictions = torch.zeros((cfg.max_phrase_num,cfg.cross_res,cfg.cross_res))
        for j in range(len(selected_nouns_clip_idx)):
            if len(selected_nouns_clip_idx[j])>1:
                weighted_attn = aggregate_cross_attention(ldm_stable,phrase_tokens[j],cross_attention,selected_nouns_clip_idx[j],tag_id,j)
            else:
                weighted_attn = cross_attention[selected_nouns_clip_idx[j][-1]]
            predictions[j] += weighted_attn
            predictions[j] = predictions[j] - predictions[j].min()
            predictions[j] =  predictions[j] / predictions[j].max()
        
        predictions = (F.interpolate(predictions[None,...],(fpn_input_data[0]["image"].shape[-2],fpn_input_data[0]['image'].shape[-1]),mode='bilinear')[0]).float().to(device)
        predictions = ImageList.from_tensors([predictions], 32).tensor
        predictions = F.interpolate(predictions, scale_factor=0.25, mode='bilinear')
        predictions = upsample_eval(predictions)
        predictions = (predictions > 0.3).float()
        gts = gts.cuda()
        predictions = predictions.cuda()
        
        # Evaluation
        for p, t, th, s in zip(predictions, gts, ann_categories, ann_types):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd = p[i]
                    _, _, instance_iou = compute_mask_IoU(pd, t[i])
                    instances_iou.append(instance_iou.cpu().item())

                    if s[i] == 1:
                        singulars_iou.append(instance_iou.cpu().item())
                    else:
                        plurals_iou.append(instance_iou.cpu().item())
                    if th[i] == 1:
                        things_iou.append(instance_iou.cpu().item())
                    else:
                        stuff_iou.append(instance_iou.cpu().item())

    # # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if dist.get_rank()==0:
        print('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                            accuracy,
                                            AA,
                                            AA_singulars,
                                            AA_plurals,
                                            AA_things,
                                            AA_stuff))

def load_cross_attention(tokenizer,splited_tokens,tag_id,res=16):
    cross_attention = []
    tokenizer.model_max_length=230
    for i in range(len(splited_tokens)):
        cur_p_cross_attn = torch.load(f'./outputs/attn_db/{tag_id}/cross{res}_{i}.pt')
        enc_inp = tokenizer(
            splited_tokens[i],
            max_length=230,
        ).input_ids
        decoder = tokenizer.decode
        tokens =[]
        for k in enc_inp:
            tokens+=k[1:-1]
        selected_cross_attn = cur_p_cross_attn[...,1:len(tokens)+1]
        cross_attention.append(selected_cross_attn)
    cross_attention = torch.concat(cross_attention,dim=-1).permute(2,0,1)
    return cross_attention

def sam_refine_mask(sam_proposal_masks,mask,beta=0.3,tao=0.5):
    refine_masks = torch.zeros_like(mask).cuda()
    cur_pred = (mask>beta).float()
    cnt = 0 
    pseudo_part = cur_pred
    pseudo_part = torch.tensor(pseudo_part).cuda()
    for t in range(sam_proposal_masks.shape[0]):
        _foreground = (sam_proposal_masks[t]).float()
        if _foreground.dim()==3:
            _foreground = _foreground[0]
        # if _foreground.sum()<10:
        #     continue
        inter_1 = (_foreground * pseudo_part).sum()/(_foreground.sum())
        inter_2 = (_foreground * pseudo_part).sum()/(pseudo_part.sum()+1e-9)
        if inter_1 > tao or inter_2 > tao:
            refine_masks[_foreground.bool()] = 1
            cnt +=1
    if cnt ==0:
        refine_masks = cur_pred
    return refine_masks

def self_enhanced_fun(self_attn,cross_attn_ori,res,densecrf=False,img=None,beta=0.4):
    if self_attn.size()<cross_attn_ori.size():
        self_attn = F.interpolate(self_attn.reshape(1,1,self_attn.shape[0]**2,self_attn.shape[0]**2),size=(res**2,res**2),mode='bilinear').reshape(res,res,res,res)
    valid_points_y,valid_points_x = torch.where(cross_attn_ori>beta)
    avg_self_attn = torch.zeros_like(cross_attn_ori)
    for y,x in zip(valid_points_y,valid_points_x):
        tmp = self_attn[int(y),int(x)]
        # tmp = tmp-tmp.min()
        # tmp = tmp/tmp.max()
        avg_self_attn+=tmp
    avg_self_attn = avg_self_attn - avg_self_attn.min()
    avg_self_attn = avg_self_attn / avg_self_attn.max()

    return avg_self_attn

def dist_evaluate_self_enhanced(cfg,data_loader,device):
    scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
    ldm_stable =  StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler)
    tokenizer = CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='tokenizer')
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    cnt = 0
    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []

    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector,fpn_input_data,tag_id)) in tqdm(enumerate(data_loader)):
        ann_categories = ann_categories
        ann_types = ann_types
        image_id = fpn_input_data[0]['image_id']
        tag_id = tag_id.item()
        grad_sample = ann_types != 0
        enc_inp = list(tokenizer(
                caption[0],
                max_length=230,
                return_tensors="pt",
        ).input_ids.numpy())[0]
        decoder = tokenizer.decode
        tokens = [ decoder(k) for k in enc_inp[1:-1]]
        bert_token_ids = torch.tensor(bert_tokenizer(caption,max_length=230)['input_ids'])
        words_list =  [bert_tokenizer.decode(n) for n in list(bert_token_ids.reshape(-1,1).numpy())][1:-1]

        phrase_list = []
        valid_phrase_idx =[]            
        valid_noun_vector = noun_vector_padding[0][1:len(bert_token_ids[0])-1]
        cur_phrase = None
        valid_phrase_bert_token_ids = []
        cur_phrase_bert_token_ids = []

        k = 0 
        tokens_length = len(words_list)
        while k<tokens_length:
            if k<tokens_length-1:
                if cur_phrase is None:
                    cur_phrase = words_list[k]
                else:
                    cur_phrase = cur_phrase + ' '+ words_list[k]
                cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
            elif k==tokens_length-1:
                if valid_noun_vector[k].item()!=valid_noun_vector[k-1].item():
                    phrase_list.append(words_list[-1])
                    valid_phrase_idx.append(valid_noun_vector[k].item())
                    valid_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
                else:
                    cur_phrase = cur_phrase + ' '+ words_list[k]
                    cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())

                    phrase_list.append(cur_phrase)
                    valid_phrase_idx.append(valid_noun_vector[k].item())
                    valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
                    cur_phrase = None
                    cur_phrase_bert_token_ids = []
            if k< tokens_length-1 and valid_noun_vector[k]!=valid_noun_vector[k+1]:
                valid_phrase_idx.append(valid_noun_vector[k].item())
                phrase_list.append(cur_phrase)
                valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
                cur_phrase = None
                cur_phrase_bert_token_ids = []
            k+=1

        cnt+=1
        phrase_list = []
        for k in valid_phrase_bert_token_ids:
            if type(k)!=list:
                k = [k]
            phrase_list.append(bert_tokenizer.decode(k))
        if len(tokens)<=75:
            splited_tokens = [tokens]
        else:
            splited_tokens = split_sentences(tokens)
        cross_attention = []
        tokenizer.model_max_length=230

        self_attn = torch.load(f'./outputs/attn_db/{tag_id}/self_{cfg.self_res}.pt')
        cross_attention = load_cross_attention(tokenizer,splited_tokens,tag_id,cfg.cross_res)

        clip_phrase_list_idx = []
        for p in phrase_list:
            if p=="'s":
                clip_phrase_list_idx.append([568])
            elif p=="' s":
                clip_phrase_list_idx.append([568])
            elif "' s" in p:
                clip_phrase_list_idx.append(tokenizer(p.replace("' s","'s"))['input_ids'][1:-1])
            else:
                clip_phrase_list_idx.append(tokenizer(p)['input_ids'][1:-1])

        selected_nouns_clip_idx = []
        phrase_tokens = []
        cum = 0
        for i in range(len(clip_phrase_list_idx)):
            tmp=[]
            tokens = []
            if valid_phrase_idx[i]>0:
                for j in range(len(clip_phrase_list_idx[i])):
                    tmp.append(cum)
                    tokens.append(decoder(clip_phrase_list_idx[i][j]))
                    cum+=1
                selected_nouns_clip_idx.append(tmp)
                phrase_tokens.append(tokens)
            else:
                cum+=len(clip_phrase_list_idx[i])


        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
            gts = upsample_eval(gts)
        
        self_attn = self_attn.reshape(cfg.self_res,cfg.self_res,cfg.self_res,cfg.self_res)
        inter_res = max(cfg.self_res,cfg.cross_res)
        predictions = torch.zeros((cfg.max_phrase_num,inter_res,inter_res))
        for j in range(len(selected_nouns_clip_idx)):
            if len(selected_nouns_clip_idx[j])>1:
                weighted_attn = aggregate_cross_attention(ldm_stable,phrase_tokens[j],cross_attention,selected_nouns_clip_idx[j],tag_id,j)
            else:
                weighted_attn = cross_attention[selected_nouns_clip_idx[j][-1]]
            weighted_attn = F.interpolate(weighted_attn[None,None,...],size=(inter_res,inter_res),mode='bilinear')[0,0]
            weighted_attn = weighted_attn - weighted_attn.min()
            weighted_attn = weighted_attn / weighted_attn.max()
            predictions[j] += self_enhanced_fun(self_attn,weighted_attn,inter_res,False,None,beta = cfg.beta)       

        predictions = (F.interpolate(predictions[None,...],(fpn_input_data[0]["image"].shape[-2],fpn_input_data[0]['image'].shape[-1]),mode='bilinear')[0]).float().to(device)
        predictions = ImageList.from_tensors([predictions], 32).tensor
        predictions = F.interpolate(predictions, scale_factor=0.25, mode='bilinear')
        predictions = upsample_eval(predictions)
        predictions = (predictions > cfg.alpha).float()

        gts = gts.cuda()
        predictions = predictions.cuda()
        
        # Evaluation
        for p, t, th, s in zip(predictions, gts, ann_categories, ann_types):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd = p[i]
                    _, _, instance_iou = compute_mask_IoU(pd, t[i])
                    instances_iou.append(instance_iou.cpu().item())

                    if s[i] == 1:
                        singulars_iou.append(instance_iou.cpu().item())
                    else:
                        plurals_iou.append(instance_iou.cpu().item())
                    if th[i] == 1:
                        things_iou.append(instance_iou.cpu().item())
                    else:
                        stuff_iou.append(instance_iou.cpu().item())

    # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if dist.get_rank()==0:
        print('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                            accuracy,
                                            AA,
                                            AA_singulars,
                                            AA_plurals,
                                            AA_things,
                                            AA_stuff))

def dist_evaluate_sam_enhanced(cfg,data_loader,device):

    scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
    ldm_stable =  StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    tokenizer = ldm_stable.tokenizer
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    cnt = 0
    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []

    for (batch_idx, (caption, grounding_instances, ann_categories, \
        ann_types, noun_vector_padding, ret_noun_vector,fpn_input_data,tag_id)) in tqdm(enumerate(data_loader)):
        ann_categories = ann_categories
        ann_types = ann_types
        image_id = fpn_input_data[0]['image_id']
        tag_id = tag_id.item()
        grad_sample = ann_types != 0
        sam_everything_masks = torch.load(f'./output/sam_db/{tag_id}.pt')
        sam_everything_masks = sam_everything_masks.cuda().float()
        # png_segments = ppmn_narr[imgid2ann[str(image_id)]]
        enc_inp = list(ldm_stable.tokenizer(
                caption[0],
                max_length=230,
                return_tensors="pt",
        ).input_ids.numpy())[0]
        decoder = ldm_stable.tokenizer.decode
        tokens = [ decoder(k) for k in enc_inp[1:-1]]
        bert_token_ids = torch.tensor(bert_tokenizer(caption,max_length=230)['input_ids'])
        words_list =  [bert_tokenizer.decode(n) for n in list(bert_token_ids.reshape(-1,1).numpy())][1:-1]

        phrase_list = []
        valid_phrase_idx =[]            
        valid_noun_vector = noun_vector_padding[0][1:len(bert_token_ids[0])-1]
        cur_phrase = None
        valid_phrase_bert_token_ids = []
        cur_phrase_bert_token_ids = []

        k = 0 
        tokens_length = len(words_list)
        while k<tokens_length:
            if k<tokens_length-1:
                if cur_phrase is None:
                    cur_phrase = words_list[k]
                else:
                    cur_phrase = cur_phrase + ' '+ words_list[k]
                cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
            elif k==tokens_length-1:
                if valid_noun_vector[k].item()!=valid_noun_vector[k-1].item():
                    phrase_list.append(words_list[-1])
                    valid_phrase_idx.append(valid_noun_vector[k].item())
                    valid_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())
                else:
                    cur_phrase = cur_phrase + ' '+ words_list[k]
                    cur_phrase_bert_token_ids.append(bert_token_ids[:,1:-1][:,k].item())

                    phrase_list.append(cur_phrase)
                    valid_phrase_idx.append(valid_noun_vector[k].item())
                    valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
                    cur_phrase = None
                    cur_phrase_bert_token_ids = []
            if k< tokens_length-1 and valid_noun_vector[k]!=valid_noun_vector[k+1]:
                valid_phrase_idx.append(valid_noun_vector[k].item())
                phrase_list.append(cur_phrase)
                valid_phrase_bert_token_ids.append(cur_phrase_bert_token_ids)
                cur_phrase = None
                cur_phrase_bert_token_ids = []
            k+=1

        cnt+=1
        phrase_list = []
        for k in valid_phrase_bert_token_ids:
            if type(k)!=list:
                k = [k]
            phrase_list.append(bert_tokenizer.decode(k))
        if len(tokens)<=75:
            splited_tokens = [tokens]
        else:
            splited_tokens = split_sentences(tokens)
        cross_attention = []
        ldm_stable.tokenizer.model_max_length=230
        self_attn = torch.load(f'./outputs/attn_db/{tag_id}/self_{cfg.self_res}.pt')
        cross_attention = load_cross_attention(ldm_stable,splited_tokens,tag_id,cfg.cross_res)

        clip_phrase_list_idx = []
        for p in phrase_list:
            if p=="'s":
                clip_phrase_list_idx.append([568])
            elif p=="' s":
                clip_phrase_list_idx.append([568])
            elif "' s" in p:
                clip_phrase_list_idx.append(tokenizer(p.replace("' s","'s"))['input_ids'][1:-1])
            else:
                clip_phrase_list_idx.append(tokenizer(p)['input_ids'][1:-1])
        
        selected_nouns_clip_idx = []
        phrase_tokens = []
        cum = 0
        for i in range(len(clip_phrase_list_idx)):
            tmp=[]
            tokens = []
            if valid_phrase_idx[i]>0:
                for j in range(len(clip_phrase_list_idx[i])):
                    tmp.append(cum)
                    tokens.append(decoder(clip_phrase_list_idx[i][j]))
                    cum+=1
                selected_nouns_clip_idx.append(tmp)
                phrase_tokens.append(tokens)
            else:
                cum+=len(clip_phrase_list_idx[i])

        with torch.no_grad():
            gts = [F.interpolate(grounding_instances[i]["gt"].unsqueeze(0), \
                                (fpn_input_data[i]['image'].shape[-2], fpn_input_data[i]['image'].shape[-1]), \
                                mode='bilinear').squeeze() for i in range(len(grounding_instances))]
            gts = ImageList.from_tensors(gts, 32).tensor
            gts = F.interpolate(gts, scale_factor=0.25, mode='bilinear')
            gts = (gts > 0).float()
            gts = upsample_eval(gts)
        
        self_attn = self_attn.reshape(cfg.self_res,cfg.self_res,cfg.self_res,cfg.self_res)
        image_path = osp.join("./datasets/coco/val2017","{:012d}.jpg".format(int(image_id)))

        h,w = fpn_input_data[0]['height'], fpn_input_data[0]['width']
        interploate_predictions = torch.zeros((cfg.max_phrase_num,h,w)).cuda()
        sam_refine_predictions = torch.zeros((cfg.max_phrase_num,h,w)).cuda()

        img = None
        predictions = torch.zeros((cfg.max_phrase_num,cfg.self_res,cfg.self_res))
        for j in range(len(selected_nouns_clip_idx)):
            if len(selected_nouns_clip_idx[j])>1:
                weighted_attn = aggregate_cross_attention(ldm_stable,phrase_tokens[j],cross_attention,selected_nouns_clip_idx[j],tag_id,j)
            else:
                weighted_attn = cross_attention[selected_nouns_clip_idx[j][-1]]
            weighted_attn = F.interpolate(weighted_attn[None,None,...],size=(cfg.self_res,cfg.self_res),mode='bilinear')[0,0]
            weighted_attn = weighted_attn - weighted_attn.min()
            weighted_attn = weighted_attn / weighted_attn.max()
            predictions[j] = self_enhanced_fun(self_attn,weighted_attn,cfg.self_res,img)
            interploate_predictions[j] = F.interpolate(predictions[j][None,None,...],size=(h,w),mode='bilinear')[0][0]
            sam_refine_predictions[j] = sam_refine_mask(sam_everything_masks,interploate_predictions[j],cfg.alpha,cfg.tao)
        
        # self enhanced
        predictions = (F.interpolate(sam_refine_predictions[None,...],(fpn_input_data[0]["image"].shape[-2],fpn_input_data[0]['image'].shape[-1]),mode='bilinear')[0]).float().to(device)
        predictions = ImageList.from_tensors([predictions], 32).tensor
        predictions = F.interpolate(predictions, scale_factor=0.25, mode='bilinear')
        predictions = upsample_eval(predictions)
        gts = gts.cuda()
        predictions = predictions.cuda()

        gts = gts.cuda()
        predictions = predictions.cuda()
        
        # Evaluation
        for p, t, th, s in zip(predictions, gts, ann_categories, ann_types):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd = p[i]
                    _, _, instance_iou = compute_mask_IoU(pd, t[i])
                    instances_iou.append(instance_iou.cpu().item())

                    if s[i] == 1:
                        singulars_iou.append(instance_iou.cpu().item())
                    else:
                        plurals_iou.append(instance_iou.cpu().item())
                    if th[i] == 1:
                        things_iou.append(instance_iou.cpu().item())
                    else:
                        stuff_iou.append(instance_iou.cpu().item())

    # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if dist.get_rank()==0:
        print('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                            accuracy,
                                            AA,
                                            AA_singulars,
                                            AA_plurals,
                                            AA_things,
                                            AA_stuff))

def load_png():
    data = json.load(open("./datasets/coco/annotations/png_coco_val2017_dataloader.json"))
    return data

def zero_shot(cfg):
    val_dataset = PanopticNarrativeGroundingValDataset(cfg,'val2017', False)
    # torch.cuda.set_device(global_rank)
    # distributed_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=default_collate,
        shuffle=False
    )
    if not cfg.self_enhanced:
        print(f"disable self enhanced--|--cross resolution [{cfg.cross_res}]---")
        dist_evaluate(cfg,val_loader,device='cpu')
    elif not cfg.sam_enhanced:
        print("self enhanced")
        print(f"self enhanced--|--cross resolution [{cfg.cross_res}]---|--self resolution[{cfg.self_res}]---|---beta [{cfg.beta}]---|---alpha [{cfg.alpha}]---")
        dist_evaluate_self_enhanced(cfg,val_loader,device='cpu')
    else:
        print("SAM enhanced")
        print(f"SAM enhanced--|--cross resolution [{cfg.cross_res}]---|--self resolution[{cfg.self_res}]---|---beta [{cfg.beta}]---|---alpha [{cfg.alpha}]---|---tao [{cfg.tao}]---")
        dist_evaluate_sam_enhanced(cfg,val_loader,device='cpu')
    
if __name__=='__main__':
    dist.init_process_group(backend="nccl", init_method='env://', world_size=-1, rank=-1, group_name='')
    
    # test
    args = parse_args()
    zero_shot(args)

