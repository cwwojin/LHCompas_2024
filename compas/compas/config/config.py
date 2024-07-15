# compas/config/config.py
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data Config
# -----------------------------------------------------------------------------

_C.DATA_PATH = "./data/custom/LH_Dataset.csv"
_C.INPUT_STEPS = 3
_C.OUTPUT_STEPS = 1
_C.TEST_SIZE = 12
_C.VAL_SIZE = 12
_C.X_COLS = None

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------

# Default : LSTM
_C.MODEL_TYPE = "lstm"

_C.LSTM = CN()
_C.LSTM.NUM_LAYERS = 1
_C.LSTM.HIDDEN_SIZE = 32
_C.LSTM.DROPOUT = 0.0
_C.LSTM.BIDIRECTIONAL = False

_C.GRU = CN()
_C.GRU.NUM_LAYERS = 1
_C.GRU.HIDDEN_SIZE = 32
_C.GRU.DROPOUT = 0.0
_C.GRU.BIDIRECTIONAL = False

_C.CNN = CN()

# -----------------------------------------------------------------------------
# Experiment Config
# -----------------------------------------------------------------------------

_C.N_EPOCHS = 50
_C.BATCH_SIZE = 8

# Logger - MLFlow, Databricks
_C.MLFLOW_TRACKING_URI = "databricks"
_C.DATABRICKS_WORKSPACE = "/Users/user"
_C.EXPERIMENT_NAME = "experiment"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
