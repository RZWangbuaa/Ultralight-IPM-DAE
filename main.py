import yaml
import logging
import argparse
import importlib
import torch
from datetime import datetime
from tools.utils import BasicDataModule
from tools.pl_callbacks import CustomProgressBar, Model_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

torch.set_float32_matmul_precision('medium')

def parser_args():
    parser = argparse.ArgumentParser(description='ecg_denoising')
    with open(file='./config.yaml', mode='r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    parser.add_argument('--use_model', type=str, required=False, default='Ultralight_IPM_DAE')
    parser.add_argument('--dataset_root_path', type=str, required=False, default='./datasets')
    parser.add_argument('--dataset_name_path', type=str, required=False, default='qt-database-1.0.0')
    parser.add_argument('--split_idx', type=int, default=91, help='dim of dataset')
    parser.add_argument('--time_win', type=int, default=5, help='ECG data length(s)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--max_epochs', type=int, default=600, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--snr', type=int, default=0, help='snr of the noisy ecg data')

    args = parser.parse_args()
    assert args.use_model in config.keys()
    args.log_dir = config[args.use_model]['log_dir']
    module = importlib.import_module(config[args.use_model]['model'])
    pl_model_ = getattr(module, 'ModelInterface_')
    pl_model = pl_model_()
    return args, pl_model


if __name__ == '__main__':
    print('\n==> Start time: ', datetime.now())
    train_times = 1
    if train_times > 1:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    for i in range(train_times):
        print('==> Train num at: ', i)
        args, pl_model = parser_args()
        ECG_loaders = BasicDataModule(args)
        progress_bar = CustomProgressBar(args, ECG_loaders.max_loader_batches)
        model_checkpoint = Model_checkpoint()
        logger = TensorBoardLogger(save_dir=args.log_dir, name="")
        start_time = datetime.now()
        ECG_trainer = Trainer(max_epochs=args.max_epochs, logger=logger, devices=1,
                              callbacks=[progress_bar, model_checkpoint],
                              num_sanity_val_steps=0)
        ECG_trainer.fit(pl_model, ECG_loaders.train_dataloader(), ECG_loaders.valid_dataloader())
        print(f"==> Training time is: {datetime.now() - start_time}")
    print('==> Completed at:', datetime.now())
    print()
