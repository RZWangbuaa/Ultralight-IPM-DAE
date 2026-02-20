from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from tools.load_qtdb import qtdb_loader


class qrdb_wrapper:
    def __init__(self, path_ecg, path_noise, batch_size=32, num_works=1, time_win=5, split_idx=91, snr=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_works = num_works
        self.train_ds = qtdb_loader(path_ecg, path_noise, get_mode='train', time_win=time_win, split_idx=split_idx, snr=snr)
        self.valid_ds = qtdb_loader(path_ecg, path_noise, get_mode='valid', time_win=time_win, split_idx=split_idx, snr=snr)

    def get_data_loaders(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_works, persistent_workers=True)
        valid_loader = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_works, persistent_workers=True)
        return train_loader, valid_loader


class BasicDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        path_ecg = Path(args.dataset_root_path) / args.dataset_name_path
        path_noise = Path('./datasets/mit-bih-noise-stress-test-database-1.0.0')
        self.dataset = qrdb_wrapper(path_ecg,
                                    path_noise,
                                    batch_size=args.batch_size,
                                    num_works=args.num_workers,
                                    time_win=args.time_win,
                                    split_idx=args.split_idx,
                                    snr=args.snr)
        self.train_loader, self.valid_loader = self.dataset.get_data_loaders()
        self.max_loader_batches = max(len(self.train_loader), len(self.valid_loader))

    def train_dataloader(self):
        return self.train_loader

    def valid_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.valid_loader
