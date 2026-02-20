import os
import tools
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm_plus import Mamba
from models.BasicModelInterface import ModelInterface


class DAE_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride):
        super().__init__()
        padding = int(ks // 2)
        self.conv = nn.Conv1d(in_channels, out_channels, ks, stride, padding)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.conv(x))


class DAE_Conv_Trans(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, outpadding=0, act=1):
        super().__init__()
        padding = int(ks // 2)
        self.conv_trans = nn.ConvTranspose1d(in_channels, out_channels, ks, stride, padding, outpadding)
        if act == 1:
            self.act = nn.ReLU(True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.conv_trans(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, inner_fc=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        ratio = 1
        if inner_fc:
            self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
                                    nn.ReLU(),
                                    nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False))
        else:
            self.fc = nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out


class attention_layer(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.temperature = 1.

    def forward(self, x):
        q = k = v = x
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output


class MF_IPM_Layer(nn.Module):
    def __init__(self, input_dim, output_dim=None, d_state=16, d_conv=4, expand=2, factor=4, device='cuda'):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.factor = factor
        self.norm = nn.LayerNorm(input_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        d_model = input_dim // factor
        d_state = d_state
        expand = expand
        d_inner = int(expand * d_model)

        self.A_list_1 = nn.ParameterList()
        for i in range(factor):
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
            self.A_list_1.append(A_log)

        self.A_list_2 = nn.ParameterList()
        for i in range(factor):
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
            self.A_list_2.append(A_log)

        use_A = True
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_external_A=use_A)
        self.mamba_r = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_external_A=use_A)

        self.down_factor = [min(2 ** i, 4) for i in range(factor)]
        self.conv_chunk = nn.ModuleList([nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=self.down_factor[i]),
            nn.ReLU(),
        ) for i in range(factor)])
        
        self.local_attn = nn.ModuleList([ChannelAttention(d_model) for _ in range(factor)])
        self.attention_layer = attention_layer(d_model)
        self.up_layers = nn.ModuleList([nn.Upsample(scale_factor=self.down_factor[i]) for i in range(factor)])
        self.d_model = d_model

    def forward(self, x):
        B, L, C = x.shape
        assert C == self.input_dim
        x = self.norm(x)
        x_list = []
        attn_list = []

        for i in range(self.factor):
            temp_x = self.conv_chunk[i](x.transpose(1, 2))  # multiscale feature extraction
            x_list.append(temp_x.transpose(1, 2))
            local_attn_map = self.local_attn[i](temp_x)
            attn_list.append(torch.squeeze(local_attn_map, dim=-1))

        mix_attn_map = torch.stack(attn_list, dim=1)
        out_attn_map = self.attention_layer(mix_attn_map)
        mod_attn_map  = torch.chunk(out_attn_map, chunks=self.factor, dim=1)
        x_list_mamba = []

        for i in range(len(x_list)):
            x_temp_mamba = self.mamba(x_list[i], A_log=self.A_list_1[i])
            x_temp_mamba_r = self.mamba_r(x_list[i].flip(dims=[1]), A_log=self.A_list_2[i])
            x_mamba = (x_temp_mamba + x_temp_mamba_r.flip(dims=[1])) * mod_attn_map[i]  # channel modulation
            x_mamba = x_mamba + x_list[i] * self.skip_scale
            x_mamba = self.up_layers[i](x_mamba.transpose(1, 2)).transpose(1, 2)
            if x_mamba.shape[1] != L:
                padding = torch.zeros([B, L - x_mamba.shape[1], self.d_model]).to(x.device)
                x_mamba = torch.cat([x_mamba, padding], dim=1)
            x_list_mamba.append(x_mamba)
        
        x_mamba = torch.cat(x_list_mamba, dim=-1)
        x_mamba = self.norm(x_mamba)
        return x_mamba


class BasicDAEmodel(nn.Module):
    def __init__(self):
        super().__init__()
        enc_ch = [[2, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
        ks = [3, 3, 3, 3, 3]
        dec_ks = [sub[::-1] for sub in enc_ch][::-1]
        factor = [1, 2, 4, 8]

        # CDAE Encoder
        self.enc_conv_1 = DAE_Conv_Block(enc_ch[0][0], enc_ch[0][1], ks=ks[0], stride=2)
        self.enc_conv_2 = DAE_Conv_Block(enc_ch[1][0], enc_ch[1][1], ks=ks[1], stride=2)
        self.enc_conv_3 = DAE_Conv_Block(enc_ch[2][0], enc_ch[2][1], ks=ks[2], stride=2)
        self.enc_conv_4 = DAE_Conv_Block(enc_ch[3][0], enc_ch[3][1], ks=ks[3], stride=2)
        self.enc_conv_5 = DAE_Conv_Block(enc_ch[4][0], enc_ch[4][1], ks=ks[4], stride=2)

        # Skip Mamba layer
        self.skip_layer_1 = MF_IPM_Layer(enc_ch[0][1], factor=factor[0])
        self.skip_layer_2 = MF_IPM_Layer(enc_ch[1][1], factor=factor[1])
        self.skip_layer_3 = MF_IPM_Layer(enc_ch[2][1], factor=factor[2])
        self.skip_layer_4 = MF_IPM_Layer(enc_ch[3][1], factor=factor[3])
        self.skip_layer_5 = nn.Identity()

        # CDAE Decoder
        self.dec_trsc_1 = DAE_Conv_Trans(dec_ks[0][0], dec_ks[0][1], ks=ks[4], stride=2, outpadding=0)
        self.dec_trsc_2 = DAE_Conv_Trans(dec_ks[1][0], dec_ks[1][1], ks=ks[3], stride=2, outpadding=0)
        self.dec_trsc_3 = DAE_Conv_Trans(dec_ks[2][0], dec_ks[2][1], ks=ks[2], stride=2, outpadding=0)
        self.dec_trsc_4 = DAE_Conv_Trans(dec_ks[3][0], dec_ks[3][1], ks=ks[1], stride=2, outpadding=0)
        self.dec_trsc_5 = DAE_Conv_Trans(dec_ks[4][0], dec_ks[4][1], ks=ks[0], stride=2, outpadding=1, act=0)

    def forward(self, x):
        x = x.transpose(1, 2)  # (N, L, C) -> (N, C, L)

        enc_1 = self.enc_conv_1(x)
        enc_2 = self.enc_conv_2(enc_1)
        enc_3 = self.enc_conv_3(enc_2)
        enc_4 = self.enc_conv_4(enc_3)
        enc_5 = self.enc_conv_5(enc_4)

        skip_1 = self.skip_layer_1(enc_1.transpose(1, 2))
        skip_2 = self.skip_layer_2(enc_2.transpose(1, 2))
        skip_3 = self.skip_layer_3(enc_3.transpose(1, 2))
        skip_4 = self.skip_layer_4(enc_4.transpose(1, 2))
        skip_5 = self.skip_layer_5(enc_5.transpose(1, 2))

        dec_1 = self.dec_trsc_1(skip_5.transpose(1, 2))
        dec_2 = self.dec_trsc_2(dec_1 + skip_4.transpose(1, 2))
        dec_3 = self.dec_trsc_3(dec_2 + skip_3.transpose(1, 2))
        dec_4 = self.dec_trsc_4(dec_3 + skip_2.transpose(1, 2))
        dec_5 = self.dec_trsc_5(dec_4 + skip_1.transpose(1, 2))

        return dec_5.transpose(1, 2)  # (N, C, L) -> (N, L, C)


class ModelInterface_(ModelInterface):
    def __init__(self):
        model = BasicDAEmodel()
        criterion = F.mse_loss
        current_py = os.path.abspath(__file__)
        super().__init__(model=model,
                         criterion=criterion,
                         current_py=current_py)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.00001)
        return opt


if __name__ == '__main__':
    torch.cuda.empty_cache()
    sample_data = torch.randn((10, 1250, 2)).to('cuda')
    sample_model = BasicDAEmodel().to('cuda')

    tools.params_and_flops(sample_data, sample_model)
    # tools.calculate_optimal_batch_size(2200, 2300, sample_model)
    # tools.calculate_throughput(200, sample_model, L=10000)
    # tools.calculate_infertime(sample_model, sample_data, repetitions=20)
