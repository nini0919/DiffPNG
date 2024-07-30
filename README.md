# DiffPNG (ECCV 2024)
The official implementation of the DiffPNG paper in PyTorch.

## Exploring Phrase-Level Grounding with Text-to-Image Diffusion Model
![image](https://github.com/nini0919/DiffPNG/assets/93698917/63b706dd-3cbd-42c8-8a47-1cb61031a994)

## News
* [2024-07-30] Code is released.

## Installation

### Requirements

- Python 3.8.18
- Numpy
- Pytorch 1.11.0
- detectron2 0.3.0

1. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```
2. cd segment-anything-third-party && pip install -e . && cd ..

3. put SAM pretrained model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth into ./segment-anything

## Datasets

1. Download the 2017 MSCOCO Dataset from its [official webpage](https://cocodataset.org/#download). You will need the train and validation splits' images and panoptic segmentations annotations.

2. Download the Panoptic Narrative Grounding Benchmark from the PNG's [project webpage](https://bcv-uniandes.github.io/panoptic-narrative-grounding/#downloads). Organize the files as follows:

```
datasets
|_coco
    |_ train2017
    |_ val2017
    |_ panoptic_stuff_train2017
    |_ panoptic_stuff_val2017
    |_annotations
        |_ png_coco_train2017.json
        |_ png_coco_val2017.json
        |_ panoptic_segmentation
        |  |_ train2017
        |  |_ val2017
        |_ panoptic_train2017.json
        |_ panoptic_val2017.json
        |_ instances_train2017.json
```

## Inference

1. generate attention map by four GPUs
    ```
        bash generate_diffusion_mask_png.sh
    ```
2. generate SAM candidate mask.
    ```
        bash generate_sam_mask_png.sh
    ```
3. evaluate on PNG dataset
    ```
        bash eval_png.sh
    ```

