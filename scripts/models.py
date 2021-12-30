import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score


class SpectogramClassifier(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        num_classes,
        loss_function=F.nll_loss,
        optimizer=torch.optim.Adam,
        learning_rate=2e-4,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.maxpool1 = torch.nn.MaxPool2d((2, 4))
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool2 = torch.nn.MaxPool2d((2, 4))
        self.flatten = nn.Flatten()

        n_sizes = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(n_sizes, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        """
        Returns size of the output tensor coming out of the convolution.
        """
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        """
        Returns features coming out of convolution block.
        """
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        return x

    def _forward(self, x):
        """
        Used for inference.
        """
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)

        return x

    def forward(self, x):
        return self._forward(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        outputs = self._forward(x)
        preds = torch.argmax(outputs, dim=1)
        loss = self.loss_function(outputs, y)
        acc = accuracy_score(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_index):
        x, y = val_batch
        outputs = self._forward(x)
        preds = torch.argmax(outputs, dim=1)
        loss = self.loss_function(outputs, y)
        acc = accuracy_score(preds, y)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self._forward(x)
        loss = F.nll_loss(outputs, y)
        preds = torch.argmax(outputs, dim=1)
        # validation metrics
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
