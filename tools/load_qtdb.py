import wfdb
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tools.load_nstdb import nstdb_loader


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    range_ = max_val - min_val + 1e-9
    normalized_data = (data - min_val) / range_
    return normalized_data


def calculate_snr(x, x_recon):
    signal_power = torch.mean(x ** 2, dim=[1, 2])
    noised_power = torch.mean((x - x_recon) ** 2, dim=[1, 2]) + 1e-12
    snr = 10 * torch.log10(signal_power / noised_power)
    return snr


def get_spilt_qtdb_dataset(data, fs, time_win=5):
    win_len = fs * time_win
    assert win_len != 0
    sample_from = 0
    sample_to = sample_from + win_len
    record_len = data.shape[0]
    split_data = []
    while sample_to <= record_len:
        temp = data[sample_from:sample_to, :]
        split_data.append(temp)
        sample_from = sample_from + win_len
        sample_to = sample_from + win_len
    split_data = np.array(split_data)
    return split_data


def get_qtdb_records(path, time_win=5, select_range=None, norm=True, save_file_name_list=False):
    if select_range is None:
        select_range = np.arange(0, 91)  # 默认选择前91条数据为训练集

    file_name_list = np.array(list(path.glob("**/*.hea")))
    if save_file_name_list:
        relative_paths = [str(f.relative_to(path_qtdb)) for f in file_name_list]
        save_path = path / 'qtdb_file_list_ubuntu.txt'
        with open(save_path, 'w') as f:
            for rel_path in relative_paths:
                f.write(rel_path + '\n')
    file_name_list = file_name_list[select_range]

    data = []
    for file_name in file_name_list:
        sigbufs, header = wfdb.rdsamp(str(file_name)[:-4])
        fs = header['fs']
        split_data = get_spilt_qtdb_dataset(sigbufs, fs, time_win)
        data.append(split_data)
    data = np.concatenate(data, axis=0)
    if norm:
        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                data[i, :, j] = min_max_normalize(data[i, :, j])
    return data


def get_noisy_data(data, noise, noise_weight, use_ch_idx=None, fix_snr=True, snr=0):
    """
    :param snr: 每段信号输入噪声信噪比设置
    :param fix_snr:
    :param noise_weight:
    :param use_ch_idx: 0 or 1 of noise
    :param data: (N1, L, C)
    :param noise: (3, N2, L, C)
    :return: noisy_data: (N1, L, C)
    """
    n1, L, C = data.shape
    n2 = noise.shape[1]

    assert L == noise.shape[2]
    assert use_ch_idx in [0, 1] if use_ch_idx is not None else True

    noise_emse = np.zeros_like(data)
    for i in range(noise.shape[0]):
        rad_idx = np.random.randint(0, n2, n1)
        if use_ch_idx is None:
            temp_add_noise = noise[i, rad_idx, :, :]
        else:
            temp_add_noise = np.repeat(np.expand_dims(noise[i, rad_idx, :, use_ch_idx], axis=-1), 2, axis=-1)
        noise_emse = noise_emse + temp_add_noise * noise_weight[i]
    if fix_snr:
        assert snr is not None
        p_signal = np.mean(data ** 2, axis=(1, 2), keepdims=True)
        p_noises = np.mean(noise_emse ** 2, axis=(1, 2), keepdims=True) + 1e-12
        noised_power_target = p_signal / (10 ** (snr / 10))
        K = np.sqrt(noised_power_target / p_noises)
        noise_emse = noise_emse * K
    noisy_data = data + noise_emse
    return noisy_data


class qtdb_loader(Dataset):
    def __init__(self, path_qtdb, path_nstdb, get_mode='train', time_win=5, norm=True, noise_weight=None,
                 split_idx=91, save_file_name_list=False, snr=0):
        super().__init__()
        nstdb = nstdb_loader(path_nstdb, resample=250, norm=True)  # norm: 对噪声数据使用 z-score 归一化
        if get_mode == 'train':
            select_range = np.arange(0, split_idx)
            self.noise = nstdb.train
            self.use_noise_ch_idx = 0
            noise_weight = [1, 1, 1] if noise_weight is None else noise_weight  # 控制不同噪声比例 不改变整体SNR
        else:
            select_range = np.arange(split_idx, 105)
            self.noise = nstdb.valid
            self.use_noise_ch_idx = 1
            noise_weight = [1, 1, 1] if noise_weight is None else noise_weight

        # 读取 QTDB 数据
        self.data = get_qtdb_records(path_qtdb,
                                     select_range=select_range,
                                     time_win=time_win,
                                     norm=norm,
                                     save_file_name_list=save_file_name_list).astype(np.float32)

        # 分割 NSTDB 数据
        self.noise = nstdb.get_split_nstdb_dataset(time_win, self.noise)

        # 信号噪声混合
        self.noisy_data = get_noisy_data(self.data, self.noise, noise_weight,
                                         use_ch_idx=self.use_noise_ch_idx,
                                         snr=snr).astype(np.float32)

        self.noisy_snr_ = calculate_snr(torch.tensor(self.data), torch.tensor(self.noisy_data))
        # print(f'==> SNR: {float(torch.mean(self.noisy_snr_)):.3f} {float(torch.std(self.noisy_snr_)):.3f}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.noisy_data[idx], self.data[idx]


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]
    path_qtdb = repo_root / 'datasets' / 'qt-database-1.0.0'
    path_nstdb = repo_root / 'datasets' / 'mit-bih-noise-stress-test-database-1.0.0'
    sample_loader_1 = qtdb_loader(path_qtdb, path_nstdb, get_mode='valid', time_win=5, split_idx=91,
                                  save_file_name_list=False)
    print('==> Dim: ', sample_loader_1.data.shape)

    import matplotlib.pyplot as plt
    sample_noisy_ecg, sample_ecg = sample_loader_1[700]
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(sample_ecg[:, 0])
    axes[1].plot(sample_ecg[:, 0])
    axes[1].plot(sample_noisy_ecg[:, 0])
    plt.show()

