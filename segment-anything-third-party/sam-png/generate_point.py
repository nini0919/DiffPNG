import json
import os
import os.path as osp
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
random.seed(0)
np.random.seed(0)

dataset = 'refcoco'
path = '/home/jjy/NICE_ydn/LAVT-RIS/anns/{0}/masks/{1}'.format(dataset,dataset)
mask_list = os.listdir(path)
point_dict = {}
for idx,p in tqdm(enumerate(mask_list),total= len(mask_list)):
    if p.endswith('.npy'):
        mask = np.load(osp.join(path,p))
        plt.imshow(mask)
        y_arr,x_arr = np.where(mask>0)
        rand_i = np.random.randint(0,len(y_arr)-1)

        input_point = [[int(x_arr[rand_i]),int(y_arr[rand_i])]]
        # plt.clf()
        # plt.imshow(mask)
        # show_points(np.array(input_point),np.array([1]),plt.gca())
        # plt.savefig('tmp.png')
        mask_id = p.split('.')[0]
        if point_dict.get( mask_id) is None:
            point_dict[mask_id] =  input_point
        else:
            print('already exist.')

with open('/home/jjy/NICE_ydn/LAVT-RIS/anns/{0}/{0}_mask2point.json'.format(dataset),'w') as f:
    json.dump(point_dict,f)