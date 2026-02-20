import os
import argparse
import torch
import logging
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
from thop import profile
from tools.load_qtdb import qtdb_loader
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch.nn.functional as F

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
torch.set_float32_matmul_precision('high')

def calculate_snr(x, x_recon):
    signal_power = torch.mean(x ** 2, dim=[1, 2])
    noised_power = torch.mean((x - x_recon) ** 2, dim=[1, 2]) + 1e-12
    snr = 10 * torch.log10(signal_power / noised_power)
    return snr

def calculate_mse(x, y, reduction='none'):
    mse = F.mse_loss(x, torch.tensor(y), reduction=reduction)
    mse_per_mcgdata = mse.mean(dim=[1, 2])
    rmse_per_mcgdata = torch.sqrt(mse_per_mcgdata)
    mse_mean = mse_per_mcgdata.mean()
    return rmse_per_mcgdata, mse_mean

def calculate_CosSim(x, x_recon):
    cos_sim = F.cosine_similarity(x, torch.tensor(x_recon), dim=1)
    cos_sim_per = torch.mean(cos_sim, dim=1)
    return cos_sim_per

def calculate_prd(x, x_recon):
    diff = x - x_recon
    diff_squared_sum = torch.sum(diff ** 2, dim=[1, 2])
    orgi_squared_sum = torch.sum(x ** 2, dim=[1, 2]) + 1e-12
    prd = torch.sqrt(diff_squared_sum / orgi_squared_sum) * 100
    return prd

def box_plot_fcn(snr_raw, snr_denoise):
    fig, axes = plt.subplots(1, 1, dpi=200)
    axes.boxplot([snr_raw, snr_denoise])
    xticks_labels = ['bef', 'aft']
    axes.set_xticklabels(xticks_labels)
    axes.grid()
    plt.show()

def test_all(model_path, test_mode='last', time_win=5, get_mode='valid', snr=0):
    repo_root = Path(__file__).resolve().parent
    path_qtdb = repo_root / 'datasets' / 'qt-database-1.0.0'
    path_nstdb = repo_root / 'datasets' / 'mit-bih-noise-stress-test-database-1.0.0'

    # load dataset
    sample_loader = qtdb_loader(path_qtdb, path_nstdb, get_mode=get_mode, time_win=time_win, noise_weight=None,
                                split_idx=91, snr=snr)
    data = sample_loader.data
    noisy_data = sample_loader.noisy_data
    test_dataloader = DataLoader(sample_loader, batch_size=10, num_workers=4, shuffle=False)

    # load model
    loss_path = model_path / 'results/train_valid_results.pth'
    if not loss_path.exists():
        raise FileNotFoundError(
            f"Missing results file: {loss_path}. Run training first or update model_path to the correct run folder."
        )
    loss = torch.load(loss_path)
    valid_loss = loss['valid_loss_epoch_avg']
    if test_mode == 'last':
        chk_path = model_path / 'checkpoints/last.ckpt'
    elif test_mode == 'best':
        chk_file = list(model_path.glob("checkpoints/valid_monitor*"))
        chk_path = str(chk_file[0])
    else:
        chk_path = model_path / 'checkpoints/last.ckpt'
    print('==> Load path', chk_path)
    model_files = list((model_path / 'model').glob("*.py"))
    if not model_files:
        raise FileNotFoundError(f"No model file found under: {model_path / 'model'}")
    model_name = model_files[0].stem
    sel_module = importlib.import_module(f"models.{model_name}")
    model = sel_module.ModelInterface_.load_from_checkpoint(chk_path)

    # calculate params and flops
    sample_data = torch.randn((1, data.shape[1], data.shape[2])).to('cuda')
    flops, params = profile(model, (sample_data,), verbose=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'==> Flops: {flops / 1e6 /sample_data.shape[0]:.6f}')
    print(f'==> Params: {num_params / 1e6:.6f}')
    print(f'==> Loss: {valid_loss[-1]:.5f}')
    print(f'==> Epoch: {len(valid_loss)}')

    # test
    ECG_trainer = Trainer(logger=False, enable_model_summary=False, enable_progress_bar=True)
    ECG_trainer.test(model, test_dataloader)
    x_recon = model.test_x_recon
    print('==> Dim of test_dataset: ', x_recon.shape)
    snr_1 = calculate_snr(torch.tensor(data), noisy_data)
    snr_2 = calculate_snr(torch.tensor(data), x_recon)
    recon_rmse, _ = calculate_mse(torch.tensor(data), x_recon)
    recon_cossim = calculate_CosSim(torch.tensor(data), x_recon)
    recon_prd = calculate_prd(torch.tensor(data), x_recon)

    print(f'==> 去噪前 SNR:     {float(torch.mean(snr_1)):.3f} ({float(torch.std(snr_1)):.3f})\n')
    print(f'==> 去噪后 SNR:     {float(torch.mean(snr_2)):.3f} ({float(torch.std(snr_2)):.3f})')
    print(f'==> 去噪后 RMSE:    {float(torch.mean(recon_rmse)):.3f} ({float(torch.std(recon_rmse)):.3f})')
    print(f'==> 去噪后 CosSim:  {float(torch.mean(recon_cossim)):.3f} ({float(torch.std(recon_cossim)):.3f})')
    print(f'==> 去噪后 PRD:\t   {float(torch.mean(recon_prd)):.3f} ({float(torch.std(recon_prd)):.3f})')
    # box_plot_fcn(snr_1, snr_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model test with selectable checkpoint path.')
    parser.add_argument('--use_model', type=str, required=True, help='Model folder name under results/.')
    parser.add_argument('--version', type=int, default=1, help='Version number under results/<model>/version_<n>.')
    parser.add_argument('--test_mode', type=str, default='best', choices=['last', 'best'], help='Checkpoint to test.')
    parser.add_argument('--time_win', type=int, default=5, help='Time window size for dataset loader.')
    parser.add_argument('--get_mode', type=str, default='valid', help='Dataset split to use.')
    parser.add_argument('--snr', type=float, default=0, help='SNR level for noise injection.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    basic_path = repo_root / 'results' / args.use_model / f'version_{args.version}'
    test_all(
        basic_path,
        test_mode=args.test_mode,
        time_win=args.time_win,
        get_mode=args.get_mode,
        snr=args.snr,
    )
