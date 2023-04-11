from yacs.config import CfgNode as CN

_C = CN()

_C.BASE_DIRECTORY = "/data_supergrover3/kats/experiments/da_hackathon/mmwhs/base_code"
_C.EXPERIMENT_NAME = ""

_C.ORACLE = False

_C.DATA = CN()
_C.DATA.NUM_WORKERS = 8
_C.DATA.IN_CHANNELS = 1
_C.DATA.OUT_CHANNELS = 5
_C.DATA.MEAN_CENTER = True
_C.DATA.STANDARDIZE = True


_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.ARCHITECTURE = ''
_C.MODEL.WEIGHT = ''
_C.MODEL.CHECKPOINT_INTERVAL = 0


_C.MODEL.UNET2D = CN()
_C.MODEL.UNET2D.FEAT_CHANNELS = 32

_C.MODEL.EFFICIENT_UNET = CN()


_C.SOLVER = CN()
_C.SOLVER.USE_AMP = False
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.BASE_LR = 0.0005
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.OPTIMIZER = 'adam'
_C.SOLVER.SCHEDULER = 'step'
_C.SOLVER.LR_DECAY_FACTOR = 0.9
_C.SOLVER.LR_MILESTONES = []
_C.SOLVER.LR_DECAY_STEPS = 20
_C.SOLVER.BATCH_SIZE_TRAIN = 16
_C.SOLVER.BATCH_SIZE_TEST = 16


def get_cfg_defaults():
    return _C.clone()