export CUDA_VISIBLE_DEVICES=0
FEAT_DIM=256
python train_net.py \
    --eval-only \
    --num-gpus 1 \
    --config-file configs/ZeroPlaneParallelDomain/dust3r_large_dpt_bs16_50ep.yaml \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    OUTPUT_DIR ./visualizations/parallel_domain_test_vis \
    TEST.NO_VIS "True"
