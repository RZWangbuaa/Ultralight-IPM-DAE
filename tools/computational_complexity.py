import torch
import numpy as np
from thop import profile
from tqdm import tqdm


def params_and_flops(sample_data, sample_model):
    num_params = sum(p.numel() for p in sample_model.parameters())
    flops, params = profile(sample_model, (sample_data,), verbose=False)
    print(f'==> Flops: \t {flops / sample_data.shape[0] / 1e6:.6f}')
    print(f'==> Params:\t {num_params / 1e6:.6f}')

def calculate_optimal_batch_size(start_idx, end_idx, sample_model, L=1250, C=2):
    for i in tqdm(range(start_idx, end_idx)):
        try:
            sample_data = torch.randn((i, L, C)).to('cuda')
            sample_model(sample_data)
            del sample_data
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"CUDA OOM at batch size: {i}")
                print(f"Maximum batch size without OOM: {i-1}")
                break
            else:
                raise e

def calculate_throughput(optimal_batch_size, sample_model, L=1250, C=2):
    total_time = 0
    repetitionis = 10
    dummy_input = torch.randn(optimal_batch_size, L, C, dtype=torch.float).to('cuda')
    with torch.no_grad():
        for _ in tqdm(range(repetitionis)):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            sample_model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (repetitionis * optimal_batch_size) / total_time
    print(f'==> Throughput:\t {throughput:.6f}')

def calculate_infertime(sample_model, sample_data, repetitions=300):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    for _ in range(10):
        _ = sample_model(sample_data)
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = sample_model(sample_data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    print(f'==> Inference time:\t {mean_syn:.6f}')
