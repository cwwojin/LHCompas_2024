import os
import os.path as path
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import mlflow
from datetime import datetime
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

from compas.data_providers import TSMultiDataModule
from compas.trainers import *
from compas.config import get_cfg_defaults


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", required=False, default="train", help="one of `train`, `test`"
    )
    parser.add_argument("--config", required=True, help="path to the .yaml config file")

    return parser.parse_args()


def load_model(cfg, n_features, x_cols, no_val, scaler):
    model_type = cfg.MODEL_TYPE
    if model_type == "lstm":
        trainer = LSTMSimpleLightningModule(
            cfg=dict(
                input_size=n_features,
                input_steps=cfg.INPUT_STEPS,
                output_steps=cfg.OUTPUT_STEPS,
                num_layers=cfg.LSTM.NUM_LAYERS,
                hidden_size=cfg.LSTM.HIDDEN_SIZE,
                dropout=cfg.LSTM.DROPOUT,
                bidirectional=cfg.LSTM.BIDIRECTIONAL,
                x_cols=x_cols,
            ),
            scaler=scaler,
            no_val=no_val,
        )
    elif model_type == "gru":
        trainer = GRUSimpleLightningModule(
            cfg=dict(
                input_size=n_features,
                input_steps=cfg.INPUT_STEPS,
                output_steps=cfg.OUTPUT_STEPS,
                num_layers=cfg.GRU.NUM_LAYERS,
                hidden_size=cfg.GRU.HIDDEN_SIZE,
                dropout=cfg.GRU.DROPOUT,
                bidirectional=cfg.GRU.BIDIRECTIONAL,
                x_cols=x_cols,
            ),
            scaler=scaler,
            no_val=no_val,
        )
    elif model_type == "dlinear":
        trainer = DLinearLightningModule(
            cfg=dict(
                input_steps=cfg.INPUT_STEPS,
                output_steps=cfg.OUTPUT_STEPS,
                channels=n_features,
                kernel_size=cfg.DLINEAR.KERNEL_SIZE,
                individual=cfg.DLINEAR.INDIVIDUAL,
                x_cols=x_cols,
            ),
            scaler=scaler,
            no_val=no_val,
        )
    elif model_type == "cnn":
        trainer = CNN1DSimpleLightningModule(
            cfg=dict(
                in_channels=n_features,
                input_steps=cfg.INPUT_STEPS,
                output_steps=cfg.OUTPUT_STEPS,
                hidden_size=cfg.CNN.HIDDEN_SIZE,
                kernel_size=cfg.CNN.KERNEL_SIZE,
                dropout=cfg.CNN.DROPOUT,
                activation=cfg.CNN.ACTIVATION,
                x_cols=x_cols,
            ),
            scaler=scaler,
            no_val=no_val,
        )
    else:
        raise NotImplementedError(
            "supported model types are : 'lstm', 'gru', 'dlinear', 'cnn'"
        )
    return trainer


def get_model_name(cfg, timestamp):
    if cfg.MODEL_TYPE == "lstm":
        model_name = f"LSTM_{'bi' if cfg.LSTM.BIDIRECTIONAL else 'uni'}_in_{cfg.INPUT_STEPS}_out_{cfg.OUTPUT_STEPS}_{timestamp}"
    elif cfg.MODEL_TYPE == "gru":
        model_name = f"GRU_{'bi' if cfg.LSTM.BIDIRECTIONAL else 'uni'}_in_{cfg.INPUT_STEPS}_out_{cfg.OUTPUT_STEPS}_{timestamp}"
    elif cfg.MODEL_TYPE == "dlinear":
        model_name = f"DLinear_{'indiv' if cfg.DLINEAR.INDIVIDUAL else 'base'}_in_{cfg.INPUT_STEPS}_out_{cfg.OUTPUT_STEPS}_{timestamp}"
    elif cfg.MODEL_TYPE == "cnn":
        model_name = f"CNN_in_{cfg.INPUT_STEPS}_out_{cfg.OUTPUT_STEPS}_{timestamp}"
    else:
        raise NotImplementedError(
            "supported model types are : 'lstm', 'gru', 'dlinear', 'cnn'"
        )
    if cfg.RUN_NAME:
        model_name = f"{cfg.RUN_NAME}_{model_name}"
    return model_name


def run_experiment(args: dict):
    # load config from .yaml file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args["config"])

    # Prepare Dataset
    data_module = TSMultiDataModule(
        data_path=cfg.DATA_PATH,
        input_steps=cfg.INPUT_STEPS,
        output_steps=cfg.OUTPUT_STEPS,
        test_size=cfg.TEST_SIZE,
        val_size=cfg.VAL_SIZE,
        batch_size=cfg.BATCH_SIZE,
        x_cols=cfg.X_COLS,
    )
    data_module.prepare_data()
    data_module.setup()

    # Prepare model
    NO_VAL = cfg.VAL_SIZE <= 0
    model = load_model(
        cfg=cfg,
        n_features=data_module.n_features,
        x_cols=data_module.x_cols,
        no_val=NO_VAL,
        scaler=data_module.scaler,
    )

    # Setup Trainer & MLFlow Logger
    if cfg.MLFLOW_TRACKING_URI == "databricks":
        mlflow.login()
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%m-%s")

    trainer = pl.Trainer(
        max_epochs=cfg.N_EPOCHS,
        devices="auto",
        logger=MLFlowLogger(
            experiment_name=f"{cfg.DATABRICKS_WORKSPACE}/{cfg.EXPERIMENT_NAME}",
            run_name=(
                f"{cfg.RUN_NAME}_{timestamp}" if cfg.RUN_NAME else f"run_{timestamp}"
            ),
            tracking_uri=cfg.MLFLOW_TRACKING_URI,
            log_model=True,
        ),
        check_val_every_n_epoch=1,
        default_root_dir=".logs/",
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
        ],
        # skip validation if VAL_SIZE == 0
        limit_val_batches=0 if NO_VAL else None,
        num_sanity_val_steps=0 if NO_VAL else None,
    )

    # Train
    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Test
    trainer.test(
        model=model,
        datamodule=data_module,
    )

    # Export to TorchScript
    model_name = get_model_name(cfg, timestamp=timestamp)
    scripted_model = torch.jit.script(model.model)
    export_path = f".saved_models/{model_name}"
    if not path.isdir(export_path):
        os.makedirs(export_path, exist_ok=True)
    torch.jit.save(
        scripted_model,
        f"{export_path}/model.pt",
    )

    # Export data scaler as pickle
    with open(f"{export_path}/scaler.pkl", "wb") as f:
        pickle.dump(model.scaler, f)

    # END
    print("Experiment Done Successfully.")


if __name__ == "__main__":
    args = get_args()
    run_experiment(args=vars(args))
