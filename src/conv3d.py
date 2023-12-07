import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

from . import config


class CNN3DModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        loss_weights=torch.tensor([0.5, 0.5]),
        input_shape=(1, 120, 120, 5),
    ):
        super(CNN3DModel, self).__init__()
        self.lr = learning_rate 
        self.loss_weights = loss_weights
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(
            1, 32, kernel_size=(2, 2, 2), stride=1, padding=(1, 1, 1)
        )
        self.conv2 = nn.Conv3d(
            32, 64, kernel_size=(2, 2, 2), stride=1, padding=(1, 1, 1)
        )

        self.conv1_bn = nn.BatchNorm3d(32)
        self.conv2_bn = nn.BatchNorm3d(64)

        self.fc1_bn = nn.BatchNorm1d(128)

        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.3)

        # Calculating the size of the input for the fully connected layer
        dummy_input = torch.randn(1, *input_shape)
        dummy_output = self._forward_conv(dummy_input)
        self.fc1_input_shape = dummy_output.view(-1).shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_shape, 128)
        self.fc2 = nn.Linear(128, 2)

        # Metrics
        self.accuracy = Accuracy(task="binary", num_classes=2)
        self.precision = Precision(num_classes=2, average="weighted", task="BINARY")
        self.recall = Recall(num_classes=2, average="weighted", task="BINARY")
        self.f1_score = F1Score(task="binary", num_classes=2)

    def _forward_conv(self, x):
        x = x.to(torch.float32)
        
        x = self.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = F.max_pool3d(x, (3, 3, 3))
        
        x = self.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = F.max_pool3d(x, (3, 3, 3))

        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)

        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y, weight=self.loss_weights)
        y_hat = torch.argmax(torch.sigmoid(logits), dim=1)
        return loss, logits, y_hat, y

    def training_step(self, batch, batch_idx):
        # Forward pass
        loss, logits, y_hat, y = self._common_step(batch, batch_idx)

        # # Compute metrics
        acc = self.accuracy(y_hat, y)
        prec = self.precision(y_hat, y)
        rec = self.recall(y_hat, y)
        f1 = self.f1_score(y_hat, y)

        # Log metrics
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": acc,
                "train_precision": prec,
                "train_recall": rec,
                "train_f1": f1,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Forward pass
        loss, logits, y_hat, y = self._common_step(batch, batch_idx)

        # Compute metrics
        acc = self.accuracy(y_hat, y)
        prec = self.precision(y_hat, y)
        rec = self.recall(y_hat, y)
        f1 = self.f1_score(y_hat, y)

        # Log metrics
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": acc,
                "val_precision": prec,
                "val_recall": rec,
                "val_f1": f1,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, logits, y_hat, y = self._common_step(batch, batch_idx)
        # # Compute metrics
        acc = self.accuracy(y_hat, y)
        prec = self.precision(y_hat, y)
        rec = self.recall(y_hat, y)
        f1 = self.f1_score(y_hat, y)

        # Log metrics
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": acc,
                "test_precision": prec,
                "test_recall": rec,
                "test_f1": f1,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
