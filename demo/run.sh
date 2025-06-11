CUDA_VISIBLE_DEVICES=0 python demo/demo.py \
    --config-file configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
    --input ./demo/0_d2_image.png \
    --out ./demo/nyu_demo \
    --opts \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    MODEL.MASK_FORMER.LEARN_NORMAL_CLS "True" \
    MODEL.MASK_FORMER.LEARN_OFFSET_CLS "True" \
    MODEL.MASK_FORMER.MIX_ANCHOR "True" \
    MODEL.MASK_FORMER.NORMAL_CLS_NUM 7 \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH "True" \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_NORMAL_ATTENTION "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_DEPTH_ATTENTION "True" \
    MODEL.MASK_FORMER.SEPARATE_PIXEL_ATTENTION "True" \
