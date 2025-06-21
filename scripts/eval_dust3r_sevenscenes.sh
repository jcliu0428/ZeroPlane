export CUDA_VISIBLE_DEVICES=0
python train_net.py \
    --eval-only \
    --num-gpus 1 \
    --config-file configs/ZeroPlaneSevenScenes/dust3r_large_dpt_bs16_50ep.yaml \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    OUTPUT_DIR ./visualizations/sevenscenes_test_vis \
    INPUT.LARGE_RESOLUTION_INPUT "False" \
    INPUT.LARGE_RESOLUTION_EVAL "False" \
    TEST.NO_VIS "True"
