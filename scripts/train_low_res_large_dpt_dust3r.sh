export CUDA_VISIBLE_DEVICES=0,1
python train_net.py \
    --num-gpus 2 \
    --config-file configs/ZeroPlaneMixed/dust3r_large_dpt_bs16_50ep.yaml \
    --dist-url 'tcp://127.0.0.1:64750' \
    OUTPUT_DIR checkpoints/dust3r_large_dpt_train_low_res \
    INPUT.DINO_INPUT_HEIGHT 196 \
    INPUT.DINO_INPUT_WIDTH 252 \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.STEPS '(40000, 47000)' \
    SOLVER.CHECKPOINT_PERIOD 10000 \
    TEST.EVAL_PERIOD 50001 \
    INPUT.LARGE_RESOLUTION_INPUT "False" \
