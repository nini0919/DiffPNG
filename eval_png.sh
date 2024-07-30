CUDA_VISABLE_DEVICES=0 python -m torch.distributed.launch --master_port 2321  --nproc_per_node=1  --nnodes=1 zero_shot_diffpng.py \
    --sam_enhanced True \
    --cross_res 16 \
    --self_res 32 \
    --beta 0.4 \
    --alpha 0.3 \
    --tao 0.6