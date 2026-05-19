import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

class DetectorLit(pl.LightningModule):
    def __init__(self, model, lr, pos_weight=3.0):
        super().__init__()
        self.model = model
        self.lr = lr
        # Use BCEWithLogitsLoss – combines sigmoid + BCE, supports pos_weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        # Return raw logits (no sigmoid)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # For accuracy, we need probabilities
        probs = torch.sigmoid(logits)
        acc = self.accuracy(probs, y)
        self.log("det/train_loss", loss, prog_bar=True)
        self.log("det/train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        acc = self.accuracy(probs, y)
        self.log("det/val_loss", loss, prog_bar=True)
        self.log("det/val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)