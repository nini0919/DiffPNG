from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import os.path as osp
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import cv2
import numpy as np
import itertools
import transformers
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    


dist.init_process_group(backend="nccl", init_method='env://', world_size=-1, rank=-1, group_name='')
world_size = dist.get_world_size()
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
sam_checkpoint = "/media/disk2/ydn/lavt/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)

# setup_for_distributed(is_main_process())
data_dir = '/media/disk2/ydn/prompt-to-prompt/outputs/p2p_nulltext_png'
png_data = json.load(open('/media/disk2/ydn/prompt-to-prompt/ppmn_narr_list.json'))
ldm_stable =  StableDiffusionPipeline.from_pretrained(
    "/media/disk2/ydn/prompt-to-prompt/AI-ModelScope/stable-diffusion-v1-4")
tokenizer = ldm_stable.tokenizer
bert_tokenizer = transformers.BertTokenizer.from_pretrained('/home/jjy/.cache/huggingface/hub/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076/')



for i in range(len(png_data)):
    # df_ts = torch.load(osp.join(data_dir,file_list[i]))
    tag_id = png_data[i]['tag_id']
    image_id = png_data[i]['image_id']
    image_path = osp.join("/home/jjy/NICE_ydn/PPMN/datasets/coco/val2017","{:012d}.jpg".format(int(image_id)))

    df_ts = torch.load(osp.join(data_dir,f'{tag_id}.pt')).permute(2,0,1)
    caption = png_data[i]['caption']
    valid_noun_vector = torch.tensor(png_data[i]['noun_vector'])
    noun_vector = torch.tensor(png_data[i]['noun_vector']).nonzero()
    enc_inp = list(ldm_stable.tokenizer(
            caption,
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
    ).input_ids.numpy())[0]

    bert_token_ids = torch.tensor(bert_tokenizer(caption,max_length=230)['input_ids'])[None,...]
    pad_bert_token_ids = F.pad(bert_token_ids,(0,230-bert_token_ids.shape[1]))
    noun_tokens_ids = pad_bert_token_ids[0,noun_vector]
    phrase_list = []
    valid_phrase_idx =[]   
    words_list =  [bert_tokenizer.decode(n) for n in list(bert_token_ids.reshape(-1,1).numpy())][1:-1]

    cur_phrase = None
    k = 0 
    tokens_length = len(words_list)
    while k<tokens_length:
        if k<tokens_length-1:
            if cur_phrase is None:
                cur_phrase = words_list[k]
            else:
                cur_phrase = cur_phrase + ' '+ words_list[k]
        elif k==tokens_length-1:
            phrase_list.append(words_list[-1])
            valid_phrase_idx.append(valid_noun_vector[k].item())
        if k< tokens_length-1 and valid_noun_vector[k]!=valid_noun_vector[k+1]:
            valid_phrase_idx.append(valid_noun_vector[k].item())
            phrase_list.append(cur_phrase)
            cur_phrase = None
        k+=1
    clip_phrase_list_idx = []
    valid_phrase = []

    for i,p in enumerate(phrase_list):
        if valid_phrase_idx[i]>0:
            valid_phrase.append(p)
        clip_phrase_list_idx.append(tokenizer(p)['input_ids'][1:-1])
        selected_nouns_clip_idx = []
        cum = 0

    for i in range(len(clip_phrase_list_idx)):
        tmp=[]
        if valid_phrase_idx[i]>0:
            for j in range(len(clip_phrase_list_idx[i])):
                tmp.append(cum)
                cum+=1
            selected_nouns_clip_idx.append(tmp)
        else:
            cum+=len(clip_phrase_list_idx[i])

    image = cv2.imread(image_path)
    h,w,c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    clip_tokens_ids = list(itertools.chain.from_iterable(clip_phrase_list_idx))

    for j in range(len(valid_phrase)):
        for k in range(len(selected_nouns_clip_idx[j])):
            cross_attn = df_ts[selected_nouns_clip_idx[j][k]+1]
            cross_attn = cross_attn - cross_attn.min()
            cross_attn = cross_attn / cross_attn.max()
            cross_attn = F.interpolate(cross_attn[None,None,...],size=(h,w),mode='bilinear')[0,0]
                        # 找到前五个最大值及其索引
            
            top_values, top_indices = torch.topk(cross_attn.flatten(), k=1)

            # 将一维索引转换为二维索引
            row_indices = top_indices // cross_attn.size(0)
            col_indices = top_indices % cross_attn.size(1)
            top_pixel_positions = torch.stack((row_indices, col_indices), dim=1)
            plt.subplot(121)
            plt.imshow(image)
            show_points(top_pixel_positions ,np.ones(1), plt.gca())
            print(tokenizer.decode(clip_tokens_ids[selected_nouns_clip_idx[j][k]]))
            masks, scores, logits = predictor.predict(
                point_coords=top_pixel_positions.cpu().numpy(),
                point_labels=np.ones(1),
                multimask_output=False,
            )
            # torch.save(torch.tensor(masks[0]),f'/media/disk2/ydn/prompt-to-prompt/outputs/p2p_nulltext_png_sam_refine/{tag_id}_{selected_nouns_clip_idx[j][k]}.pt')
            plt.subplot(122)
            plt.imshow(masks[0].astype(np.float32))
            plt.savefig('tmp.png')
            plt.clf()

            print()