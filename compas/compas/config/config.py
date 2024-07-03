# compas/config/config.py
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
# mandatory parameters : model dimensions
_C.MODEL.C_IN = None
_C.MODEL.C_OUT = None
_C.MODEL.SEQ_LEN = None
_C.MODEL.PRED_DIM = None
# model hyperparameters
_C.MODEL.N_LAYERS = 2
_C.MODEL.N_HEADS = 8
_C.MODEL.D_MODEL = 512
_C.MODEL.D_FF = 2048
_C.MODEL.DROPOUT = 0.05
_C.MODEL.ATTN_DROPOUT = 0.0
_C.MODEL.PATCH_LEN = 16
_C.MODEL.STRIDE = 8
_C.MODEL.PADDING_PATCH = True
_C.MODEL.REVIN = True
_C.MODEL.AFFINE = False
_C.MODEL.INDIVIDUAL = False
_C.MODEL.SUBTRACT_LAST = False
_C.MODEL.DECOMPOSITION = False
_C.MODEL.KERNEL_SIZE = 25
_C.MODEL.ACTIVATION = "gelu"
_C.MODEL.NORM = "BatchNorm"
_C.MODEL.PRE_NORM = False
_C.MODEL.RES_ATTENTION = True
_C.MODEL.STORE_ATTN = False


# -----------------------------------------------------------------------------
# Experiment Config
# -----------------------------------------------------------------------------


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
