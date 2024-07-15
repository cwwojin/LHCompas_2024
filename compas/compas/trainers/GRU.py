import os
import os.path as path
import pickle
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch.optim as optim

from compas.models import GRUSimple


# Lightning module - GRU
class GRUSimpleLightningModule(pl.LightningModule):
    def __init__(self, model=None, cfg=None, scaler=None, no_val=False):
        super(GRUSimpleLightningModule, self).__init__()
        assert (model is not None) or (cfg is not None)

        # init by either model or CfgNode
        if model is not None:
            self.model = model
        else:
            assert scaler is not None, "Dataset Scaler must be provided with CfgNode"
            self.model = GRUSimple(
                input_size=cfg["input_size"],
                input_steps=cfg["input_steps"],
                output_steps=cfg["output_steps"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                bidirectional=cfg["bidirectional"] if "bidirectional" in cfg else False,
                scaler=scaler,
                feature_names=cfg["x_cols"],
            )

        self.scaler = self.model.scaler
        self.criterion = nn.MSELoss()
        self.test_predictions = []
        self.no_val = no_val
        self.x_cols = self.model.feature_names

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        params = dict(
            input_size=self.model.input_size,
            input_steps=self.model.input_steps,
            output_steps=self.model.output_steps,
            output_size=self.model.output_size,
            num_layers=self.model.num_layers,
            hidden_size=self.model.hidden_size,
            dropout=self.model.dropout,
            bidirectional=self.model.bidirectional,
            criterion="mse",
            use_validation=not self.no_val,
            feature_names=self.x_cols,
        )
        self.logger.log_hyperparams(params=params)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log_dict(
            {
                "train_mse_loss": loss,
            },
            on_epoch=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log_dict(
            {
                "val_mse_loss": loss,
            },
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss}

    def on_train_end(self):
        # 1. cache the data scaler locally
        cache_path = path.join(self.trainer.default_root_dir, "scaler.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # 2. upload scaler as MLFlow artifact
        if isinstance(self.logger, MLFlowLogger):
            artifact_path = f"model/scaler"
            self.logger.experiment.log_artifact(
                self.logger.run_id,
                cache_path,
                artifact_path,
            )

        # 3. delete cached scaler
        if path.isfile(cache_path):
            os.remove(cache_path)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_predictions.append((y_hat, y))

        self.log_dict(
            {
                "test_mse_loss": loss,
            },
            on_step=True,
        )
        return {"loss": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self(x)

        # Reshape y_hat as time series (B, output_steps, input_size)
        batch_size = x.size(0)
        preds = y_hat.view(batch_size, self.model.output_steps, self.model.input_size)

        return preds

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=2,
                min_lr=1e-6,
            ),
            "monitor": "train_mse_loss_epoch" if self.no_val else "val_mse_loss",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
