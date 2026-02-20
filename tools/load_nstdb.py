import wfdb
import wfdb.processing
import numpy as np
from pathlib import Path


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    range_ = max_val - min_val + 1e-9
    normalized_data = (data - min_val) / range_
    return normalized_data


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data


def resample_fcn(data, fs, target_fs):
    chs = data.shape[1]
    resample_data = []
    for i in range(chs):
        temp, _ = wfdb.processing.resample_sig(data[:, i], fs, target_fs)
        resample_data.append(temp)
    resample_data = np.stack(resample_data)
    return resample_data.transpose(1, 0)


class nstdb_loader:
    def __init__(self, path, split_time_min=15, resample=250, norm=False):
        self.noise_type_list = ['bw', 'em', 'ma']
        self.path = path
        self.norm = norm
        self.fs = 360  # 原始数据采样率 若重采样则根据重采样频率调整
        self.spl_time = split_time_min
        self.resample = resample

        self.noise = self.get_nstdb()
        self.train, self.valid = self.split_nstdb_by_time()

    def get_nstdb(self):
        noise = []
        for i in self.noise_type_list:
            temp_path = self.path / i
            sigbufs, header = wfdb.rdsamp(temp_path)
            if self.resample is not None:
                sigbufs = resample_fcn(sigbufs, self.fs, self.resample)
            if self.norm:
                sigbufs = z_score_normalize(sigbufs)
            noise.append(sigbufs)
        self.fs = self.resample if self.resample is not None else self.fs
        return noise

    def split_nstdb_by_time(self):  # 前15min为训练集 后15min为测试集
        dataset_len = self.spl_time * 60 * self.fs
        train_set = []
        valid_set = []
        for i in self.noise:
            train_set.append(i[0:dataset_len, :])
            valid_set.append(i[-dataset_len:-1, :])
        return np.array(train_set), np.array(valid_set)

    def get_split_nstdb_dataset(self, time_win, noise):
        fs = self.resample if self.resample is not None else self.fs
        win_len = fs * time_win
        assert win_len != 0
        record_len = noise.shape[1]
        noise_new = []
        for i in range(noise.shape[0]):
            sample_from = 0
            sample_to = sample_from + win_len
            split_noise = []
            while sample_to <= record_len:
                temp = noise[i, sample_from:sample_to, :]
                split_noise.append(temp)
                sample_from = sample_from + win_len
                sample_to = sample_from + win_len
            split_noise = np.array(split_noise)
            noise_new.append(split_noise)
        noise_new = np.array(noise_new)
        return noise_new


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / 'datasets' / 'mit-bih-noise-stress-test-database-1.0.0'
    a = nstdb_loader(path, norm=True)
    b = a.get_split_nstdb_dataset(time_win=5, noise=a.valid)
    print(b.shape)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1)
    type_idx = 2
    nois_idx = 0
    axes[0].plot(b[type_idx, nois_idx, :, 0])
    axes[1].plot(b[type_idx, nois_idx, :, 1])
    plt.show()
