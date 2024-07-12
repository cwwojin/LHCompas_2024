import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from compas.models import LSTMSimple


# Lightning module
class LSTMSimpleLightningModule(pl.LightningModule):
    def __init__(self, model=None, cfg=None, scaler=None):
        super(LSTMSimpleLightningModule, self).__init__()
        assert (model is not None) or (cfg is not None)

        # init by either model or CfgNode
        if model is not None:
            self.model = model
        else:
            assert scaler is not None, "Dataset Scaler must be provided with CfgNode"
            self.model = LSTMSimple(
                input_size=cfg["input_size"],
                output_steps=cfg["output_steps"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                bidirectional=cfg["bidirectional"] if "bidirectional" in cfg else False,
                scaler=scaler,
            )

        self.scaler = self.model.scaler
        self.criterion = nn.MSELoss()
        self.test_predictions = []

    def forward(self, x):
        return self.model(x)

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=2,
                min_lr=1e-5,
            ),
            "monitor": "val_mse_loss",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
