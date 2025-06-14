from detectron2.config import CfgNode as CN

def add_ZeroPlane_config(cfg):
    """
    Add config for ZeroPlane.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    # cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic" #!
    cfg.INPUT.DATASET_MAPPER_NAME = "scannetv1_plane"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.RESIZE = True
    cfg.INPUT.BRIGHT_COLOR_CONTRAST = False

    cfg.INPUT.LARGE_RESOLUTION_INPUT = False
    cfg.INPUT.LARGE_RESOLUTION_EVAL = False

    cfg.INPUT.DINO_INPUT_HEIGHT = 196
    cfg.INPUT.DINO_INPUT_WIDTH = 252

    cfg.INPUT.DINO_UNCHANGED_ASPECT_RATIO = False

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.NORMALIZE_PARAM = False
    cfg.MODEL.PIXEL_DEPTH_LOSS_TYPE = "l1"

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.PARAM_L1_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PARAM_COS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.Q_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.CENTER_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PLANE_DEPTHS_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.WHOLE_DEPTH_WEIGHT = 2.0

    cfg.MODEL.MASK_FORMER.PIXEL_NORMAL_L1_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PIXEL_NORMAL_COS_WEIGHT = 5.0

    cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_DEPTH_WEIGHT = 0.5
    cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_NORMAL_L1_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_NORMAL_COS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_OFFSET_WEIGHT = 0.5

    cfg.MODEL.MASK_FORMER.INS_NORMAL_L1_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.INS_NORMAL_COS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.INS_OFFSET_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.INS_Q_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.REFINE_INS_PARAM_L1_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REFINE_INS_PARAM_COS_WEIGHT = 5.0

    cfg.MODEL.MASK_FORMER.LOSS_TERM_UNCERT = False

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8 # !
    # cfg.MODEL.MASK_FORMER.NHEADS = 4
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 64
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 20

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MASK_FORMER.TEST.PLANE_MASK_THRESHOLD = 0.5

    cfg.MODEL.MASK_FORMER.TEST.TEST_DEPTH_WITH_PIXEL_NORMAL = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 64
    # pixel depth decoder config
    cfg.MODEL.SEM_SEG_HEAD.DEPTH_DIM = 65
    # pixel normal decoder config
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_NORMAL_DIM = 256
    # offset decoder config
    cfg.MODEL.SEM_SEG_HEAD.OFFSET_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.MODEL.DINOv2 = CN()
    cfg.MODEL.DINOv2.ENCODER = 'vits'
    cfg.MODEL.DINOv2.NCLASS = 1
    cfg.MODEL.DINOv2.FEATURES = 64
    cfg.MODEL.DINOv2.OUT_CHANNELS = [256, 512, 1024, 1024]
    cfg.MODEL.DINOv2.USE_BN = False
    cfg.MODEL.DINOv2.USE_CLSTOKEN = False
    cfg.MODEL.DINOv2.LOCALHUB = True
    cfg.MODEL.DINOv2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DINOv2.FREEZE_BACKBONE = False
    cfg.MODEL.DINOv2.LOAD_DAv1 = False
    cfg.MODEL.DINOv2.LOAD_DAv2 = False

    cfg.MODEL.DUST3R = CN()
    cfg.MODEL.DUST3R.MODEL_NAME = 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
    cfg.MODEL.DUST3R.FEATURES = 256
    cfg.MODEL.DUST3R.OUT_CHANNELS = [256, 512, 1024, 1024]
    cfg.MODEL.DUST3R.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    cfg.MODEL.convnext = CN()
    cfg.MODEL.convnext.MODEL_TYPE = 'base'
    cfg.MODEL.convnext.IN_CHANNELS = 3
    cfg.MODEL.convnext.NCLASS = 1000
    cfg.MODEL.convnext.DEPTHS = [3, 3, 27, 3]
    cfg.MODEL.convnext.OUT_CHANNELS = [128, 256, 512, 1024]
    cfg.MODEL.convnext.IN_22K = False
    cfg.MODEL.convnext.OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']

    # hrnet32
    cfg.MODEL.arch = "hrnet_32"
    cfg.MODEL.hrnet_w32 = CN()
    cfg.MODEL.hrnet_w32.PRETRAINED = './checkpoint/hrnetv2_w32_imagenet_pretrained_new.pth'
    cfg.MODEL.hrnet_w32.STAGE1 = CN()
    cfg.MODEL.hrnet_w32.STAGE1.NUM_MODULES=1
    cfg.MODEL.hrnet_w32.STAGE1.NUM_BRANCHES= 1
    cfg.MODEL.hrnet_w32.STAGE1.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE1.NUM_BLOCKS= [4]
    cfg.MODEL.hrnet_w32.STAGE1.NUM_CHANNELS= [64]
    cfg.MODEL.hrnet_w32.STAGE1.FUSE_METHOD= None
    cfg.MODEL.hrnet_w32.STAGE2 = CN()
    cfg.MODEL.hrnet_w32.STAGE2.NUM_MODULES=1
    cfg.MODEL.hrnet_w32.STAGE2.NUM_BRANCHES= 2
    cfg.MODEL.hrnet_w32.STAGE2.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE2.NUM_BLOCKS= [4,4]
    cfg.MODEL.hrnet_w32.STAGE2.NUM_CHANNELS= [32, 64]
    cfg.MODEL.hrnet_w32.STAGE2.FUSE_METHOD= None
    cfg.MODEL.hrnet_w32.STAGE3 = CN()
    cfg.MODEL.hrnet_w32.STAGE3.NUM_MODULES=4
    cfg.MODEL.hrnet_w32.STAGE3.NUM_BRANCHES= 3
    cfg.MODEL.hrnet_w32.STAGE3.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE3.NUM_BLOCKS= [4,4,4]
    cfg.MODEL.hrnet_w32.STAGE3.NUM_CHANNELS= [32,64,128]
    cfg.MODEL.hrnet_w32.STAGE3.FUSE_METHOD= None
    cfg.MODEL.hrnet_w32.STAGE4 = CN()
    cfg.MODEL.hrnet_w32.STAGE4.NUM_MODULES=3
    cfg.MODEL.hrnet_w32.STAGE4.NUM_BRANCHES= 4
    cfg.MODEL.hrnet_w32.STAGE4.BLOCK= None
    cfg.MODEL.hrnet_w32.STAGE4.NUM_BLOCKS= [4,4,4,4]
    cfg.MODEL.hrnet_w32.STAGE4.NUM_CHANNELS= [32, 64, 128, 256]
    cfg.MODEL.hrnet_w32.STAGE4.FUSE_METHOD= None
    # cfg.MODEL.hrnet_w32.WINDOW_SIZE = 7
    # cfg.MODEL.hrnet_w32.MLP_RATIO = 4.0
    # cfg.MODEL.hrnet_w32.QKV_BIAS = True
    # cfg.MODEL.hrnet_w32.QK_SCALE = None
    # cfg.MODEL.hrnet_w32.DROP_RATE = 0.0
    # cfg.MODEL.hrnet_w32.ATTN_DROP_RATE = 0.0
    # cfg.MODEL.hrnet_w32.DROP_PATH_RATE = 0.3
    # cfg.MODEL.hrnet_w32.APE = False
    # cfg.MODEL.hrnet_w32.PATCH_NORM = True
    cfg.MODEL.hrnet_w32.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    # cfg.MODEL.hrnet_32.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    # cfg.INPUT.IMAGE_SIZE = 1024 #!
    cfg.INPUT.IMAGE_SIZE = (192, 256)
    # cfg.INPUT.MIN_SCALE = 0.1
    # cfg.INPUT.MAX_SCALE = 2.0 # !
    cfg.INPUT.MIN_SCALE = 0.6
    cfg.INPUT.MAX_SCALE = 1.5

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.MODEL.MASK_FORMER.PREDICT_CENTER = False
    cfg.MODEL.MASK_FORMER.PREDICT_PARAM = False
    cfg.MODEL.MASK_FORMER.PREDICT_DEPTH = False
    cfg.MODEL.MASK_FORMER.PREDICT_PIXEL_NORMAL = False
    cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH = True
    cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL = True
    cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_OFFSET = False

    cfg.MODEL.MASK_FORMER.DEPTH_NORMAL_PRED_UPSAMPLE = False

    cfg.MODEL.MASK_FORMER.TOKENIZE_GLOBAL_PIXEL_PRED = False
    cfg.MODEL.MASK_FORMER.SEPARATE_TOKENIZE = False

    cfg.MODEL.MASK_FORMER.PIXEL_REFINE_INS_NORMAL = False
    cfg.MODEL.MASK_FORMER.PIXEL_REFINE_INS_OFFSET = False
    cfg.MODEL.MASK_FORMER.REFINE_MASK = False
    cfg.MODEL.MASK_FORMER.WITH_INS_Q_LOSS = False
    cfg.MODEL.MASK_FORMER.WITH_INS_PARAM_LOSS = False
    cfg.MODEL.MASK_FORMER.REFINE_INS_ATTACH_PIXEL = False
    cfg.MODEL.MASK_FORMER.REFINE_INS_WITH_MASK_PROB = False
    cfg.MODEL.MASK_FORMER.REFINE_INS_FEATURE = None

    cfg.MODEL.MASK_FORMER.LEARN_IMG_CLS = False
    cfg.MODEL.MASK_FORMER.DATASET_CLASS_WEIGHT = 1.0

    cfg.MODEL.MASK_FORMER.LEARN_NORMAL_CLS = True
    cfg.MODEL.MASK_FORMER.NORMAL_CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.NORMAL_RESIDUAL_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.NORMAL_CLS_NUM = 7
    cfg.MODEL.MASK_FORMER.NORMAL_SOFT_ARGMAX = False

    cfg.MODEL.MASK_FORMER.LEARN_OFFSET_CLS = True
    cfg.MODEL.MASK_FORMER.OFFSET_CLS_NUM = 20
    cfg.MODEL.MASK_FORMER.REGRESS_INS_NORMAL = False

    cfg.MODEL.MASK_FORMER.WITH_JOINT_LOSS_CLASS = False

    cfg.MODEL.MASK_FORMER.OFFSET_CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.OFFSET_RESIDUAL_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.REGRESS_INS_OFFSET = False
    cfg.MODEL.MASK_FORMER.REGRESS_INVERSE_OFFSET = False
    cfg.MODEL.MASK_FORMER.CLASSIFY_INVERSE_OFFSET = False

    cfg.MODEL.MASK_FORMER.GLOBAL_REGRESS_INVERSE_OFFSET = False

    cfg.MODEL.MASK_FORMER.CANONICAL_FOCAL_TRAINING = False

    cfg.MODEL.MASK_FORMER.USE_INDOOR_ANCHOR = False
    cfg.MODEL.MASK_FORMER.USE_OUTDOOR_ANCHOR = False
    cfg.MODEL.MASK_FORMER.MIX_ANCHOR = True
    cfg.MODEL.MASK_FORMER.USE_COUPLED_ANCHOR = False

    cfg.MODEL.MASK_FORMER.USE_PARTIAL_CLUSTER = False

    cfg.MODEL.MASK_FORMER.TRAIN_PLANE_LEVEL_DEPTH = False

    cfg.MODEL.MASK_FORMER.WITH_NONPLANAR_QUERY = False

    cfg.MODEL.MASK_FORMER.LOAD_DPT_DEPTH_NORMAL = False

    cfg.MODEL.MASK_FORMER.NORMAL_PRED_AS_SRC = False

    cfg.MODEL.MASK_FORMER.OFFSET_PRED_AS_SRC = False

    cfg.MODEL.MASK_FORMER.DEPTH_PRED_AS_SRC = False

    cfg.MODEL.MASK_FORMER.WITH_PIXEL_NORMAL_ATTENTION = True

    cfg.MODEL.MASK_FORMER.WITH_PIXEL_OFFSET_ATTENTION = False

    cfg.MODEL.MASK_FORMER.WITH_PIXEL_DEPTH_ATTENTION = True

    cfg.MODEL.MASK_FORMER.SEPARATE_PIXEL_ATTENTION = True

    cfg.MODEL.MASK_FORMER.WITH_GT_PIXEL_ATTENTION = False

    cfg.MODEL.MASK_FORMER.WITH_MASK_UNCERT = False

    cfg.MODEL.MASK_FORMER.WO_Q_LOSS = False

    cfg.MODEL.MASK_FORMER.WITH_MASK_AGGREGATED_NORMAL_LOSS = False

    cfg.MODEL.MASK_FORMER.WITH_MASK_SELF_FITTING_LOSS = False
    cfg.MODEL.MASK_FORMER.MASK_SELF_FITTING_WEIGHT = 0.5

    cfg.TEST.NO_VIS = False

    cfg.TEST.VIS_PERIOD = 10

    cfg.TEST.INFER_ONLY = False

    cfg.TEST.SAVE_PLY = True
