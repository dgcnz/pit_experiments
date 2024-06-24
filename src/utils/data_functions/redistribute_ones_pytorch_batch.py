import torch


def redistribute_ones_pytorch_batch(batch):
    reshaped_tensors = []
    for tensor in batch:
        total_ones = tensor.sum().item()
        zeros_tensor = torch.zeros_like(tensor)
        ones_flat = torch.ones(total_ones)
        zeros_flat = zeros_tensor.flatten()
        zeros_flat[:total_ones] = ones_flat
        reshaped_tensors.append(zeros_flat.reshape(tensor.shape))
    return torch.stack(reshaped_tensors)
