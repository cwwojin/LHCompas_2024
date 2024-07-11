# compas/config/config.py
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
# mandatory parameters : model dimensions

# model hyperparameters

# -----------------------------------------------------------------------------
# Experiment Config
# -----------------------------------------------------------------------------


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
