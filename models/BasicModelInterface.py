import os
import shutil
import torch
import numpy as np
import pytorch_lightning as pl
from tools.evaluate_indicators import save_train_and_valid_results


class ModelInterface(pl.LightningModule):
    def __init__(self,
                 model,
                 criterion,
                 copy_py_file=True,
                 current_py=None):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.copy_py_file = copy_py_file
        self.current_py_path = current_py

        # num of batch (train and valid)
        self.train_num_batchs = 0
        self.valid_num_batchs = 0

        self.train_loss_epoch = 0
        self.valid_loss_epoch = 0
        self.test_loss_epoch = 0

        self.train_loss = []
        self.valid_loss = []

        self.test_x_recon = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy_data, x = batch
        x_recon = self.forward(noisy_data)
        loss = self.criterion(x_recon, x)
        self.train_loss_epoch += loss.item()
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_loss_avg', self.train_loss_epoch / (batch_idx + 1), on_step=True, on_epoch=True)
        self.train_num_batchs = batch_idx
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_data, x = batch
        x_recon = self.forward(noisy_data)
        loss = self.criterion(x_recon, x)
        self.valid_loss_epoch += loss.item()
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('valid_loss_avg', self.valid_loss_epoch / (batch_idx + 1), on_step=True, on_epoch=True)
        self.valid_num_batchs = batch_idx

    def test_step(self, batch, batch_idx):
        noisy_data, x = batch
        x_recon = self.forward(noisy_data)
        loss = self.criterion(x_recon, x)
        self.test_loss_epoch += loss.item()
        self.test_x_recon.append(x_recon.cpu())
        self.log('test_loss_avg', self.test_loss_epoch / (batch_idx + 1), on_step=True, on_epoch=False)
        return loss

    def on_train_start(self) -> None:
        if self.logger and self.copy_py_file and self.current_py_path is not None:
            version_path = os.path.join(self.logger.log_dir, 'model')
            os.makedirs(version_path, exist_ok=True)
            shutil.copy(self.current_py_path, version_path)

    def on_train_epoch_end(self) -> None:
        self.log('train_loss_per_epoch', self.train_loss_epoch, on_epoch=True)
        self.train_loss.append(self.train_loss_epoch)
        self.train_loss_epoch = 0

    def on_validation_epoch_end(self) -> None:
        self.log('valid_loss_per_epoch', self.valid_loss_epoch, on_epoch=True)
        self.valid_loss.append(self.valid_loss_epoch)
        self.valid_loss_epoch = 0

    def on_train_end(self) -> None:
        if self.logger:
            results_dict = {'train_loss_epoch_avg': np.array(self.train_loss) / (self.train_num_batchs + 1),
                            'valid_loss_epoch_avg': np.array(self.valid_loss) / (self.valid_num_batchs + 1)}
            save_train_and_valid_results(results_dict, self.logger.log_dir)

    def on_test_end(self):
        x_recon = torch.cat(self.test_x_recon).numpy()
        self.test_x_recon = x_recon
