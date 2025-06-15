CUDA_VISIBLE_DEVICES=0 python demo/demo.py \
    --config-file configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
    --input ./demo/cvpr_demo.png \
    --out ./demo/demo_out \
    --resize_w 640 \
    --resize_h 480 \
    --opts \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
