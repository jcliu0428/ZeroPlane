export CUDA_VISIBLE_DEVICES=0
FEAT_DIM=256
python train_net.py \
    --eval-only \
    --num-gpus 1 \
    --config-file configs/ZeroPlaneParallelDomain/dust3r_large_dpt_bs16_50ep.yaml \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    OUTPUT_DIR ./visualizations/parallel_domain_test_vis \
    MODEL.MASK_FORMER.LEARN_NORMAL_CLS "True" \
    MODEL.MASK_FORMER.LEARN_OFFSET_CLS "True" \
    MODEL.MASK_FORMER.MIX_ANCHOR "True" \
    MODEL.MASK_FORMER.NORMAL_CLS_NUM 7 \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH "True" \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_NORMAL_ATTENTION "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_DEPTH_ATTENTION "True" \
    MODEL.MASK_FORMER.SEPARATE_PIXEL_ATTENTION "True" \
    TEST.NO_VIS "True"
