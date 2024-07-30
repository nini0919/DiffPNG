from detectron2.data.build import _test_loader_from_config
from detectron2.config import get_cfg
import os.path as osp 

def setup(cfg_path):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg

def fpn_data(arg, split):
    cfg_dir = ('/').join(arg.detectron2_cfg.split('/')[:-1])
    cfg_path = osp.join(cfg_dir, f'panoptic_fpn_R_101_3x_{split}.yaml')
    cfg = setup(cfg_path)
    data_dict = _test_loader_from_config(cfg, cfg.DATASETS.TEST[0])
    dataset = data_dict['dataset']
    mapper = data_dict['mapper']
    return dataset, mapper