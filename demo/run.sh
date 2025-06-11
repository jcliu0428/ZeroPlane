exp_dir=dust3r_large_dpt_version_mixed_training_low_res_dim256

CUDA_VISIBLE_DEVICES=0 python demo/demo.py \
    --config-file configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
    --out ./demo/nyu_demo \
    --opts \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    OUTPUT_DIR wild_data_vis/final_dust3r/nyuv2_dataset_final_test_vis \
    MODEL.MASK_FORMER.PREDICT_PARAM "False" \
    MODEL.MASK_FORMER.PREDICT_DEPTH "False" \
    MODEL.MASK_FORMER.LEARN_NORMAL_CLS "True" \
    MODEL.MASK_FORMER.LEARN_OFFSET_CLS "True" \
    MODEL.MASK_FORMER.MIX_ANCHOR "True" \
    MODEL.MASK_FORMER.NORMAL_CLS_NUM 7 \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH "True" \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_NORMAL_ATTENTION "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_DEPTH_ATTENTION "True" \
    MODEL.MASK_FORMER.SEPARATE_PIXEL_ATTENTION "True" \
    TEST.INFER_ONLY "True" \
    TEST.VIS_PERIOD 10 \
    TEST.SAVE_PLY "True" \
