import os
import os.path as path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from datetime import datetime
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

from compas.data_providers import TSMultiDataModule
from compas.trainers import LSTMSimpleLightningModule
from compas.config import get_cfg_defaults


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", required=False, default="train", help="one of `train`, `test`"
    )
    parser.add_argument("--config", required=True, help="path to the .yaml config file")

    return parser.parse_args()


def main():
    args = get_args()

    # load config from .yaml file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

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
    model = LSTMSimpleLightningModule(
        cfg=dict(
            input_size=data_module.n_features,
            output_steps=cfg.OUTPUT_STEPS,
            num_layers=cfg.LSTM.NUM_LAYERS,
            hidden_size=cfg.LSTM.HIDDEN_SIZE,
            bidirectional=cfg.LSTM.BIDIRECTIONAL,
        ),
        scaler=data_module.scaler,
        no_val=NO_VAL,
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
            run_name=f"run_{timestamp}",
            tracking_uri=cfg.MLFLOW_TRACKING_URI,
            log_model=True,
        ),
        check_val_every_n_epoch=1,
        default_root_dir=".logs/",
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
    model_name = f"LSTM_{'bi' if model.model.bidirectional else 'uni'}_in_{data_module.input_steps}_out_{data_module.output_steps}_train_{len(data_module.train)}_val_{len(data_module.val)}_{timestamp}"
    scripted_model = torch.jit.script(model.model)
    if not path.isdir(".saved_models"):
        os.makedirs(".saved_models", exist_ok=True)
    torch.jit.save(
        scripted_model,
        f".saved_models/{model_name}.pt",
    )

    # END
    print("Experiment Done Successfully.")


if __name__ == "__main__":
    main()
