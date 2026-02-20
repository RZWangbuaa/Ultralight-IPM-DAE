import os
import torch
import matplotlib.pyplot as plt


def save_train_and_valid_results(results_dict, save_path=None):

    train_loss_epoch_avg = results_dict['train_loss_epoch_avg']
    valid_loss_epoch_avg = results_dict['valid_loss_epoch_avg']

    fig_1, axes_1 = plt.subplots(2, 1)
    axes_1[0].plot(train_loss_epoch_avg)
    axes_1[1].plot(valid_loss_epoch_avg)
    axes_1[0].grid()
    axes_1[1].grid()

    if save_path is not None:
        fig_floder = os.path.join(save_path, 'results')
        os.makedirs(fig_floder, exist_ok=True)
        pth_path = os.path.join(fig_floder, 'train_valid_results.pth')
        torch.save(results_dict, pth_path)
        fig_floder = os.path.join(save_path, 'results')
        fp_1 = os.path.join(fig_floder, "train_valid_loss_epoch_avg.png")
        fig_1.savefig(fp_1)
    else:
        plt.show()
