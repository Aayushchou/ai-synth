import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):

    def __init__(self, emb_width, n_bins, commit_cost, device="cpu"):
        super().__init__()
        self.emb_width = emb_width
        self.n_bins = n_bins
        self.commit_cost = commit_cost
        self.device = device

        self.embedding = nn.Embedding(self.n_bins, self.emb_width)
        self.embedding.weight.data.uniform_(-1 / self.n_bins, 1 / self.n_bins)

    def forward(self, x):
        N, width, T = x.shape
        # input shape -- N, C, T --> reshape to N, T, C
        x = x.permute(0, 2, 1).contiguous()

        # flatten N * T, C
        flat_x = x.view(-1, x.shape[-1])
        assert flat_x.shape[-1] == self.emb_width

        # calculate cosine distances C-DIST
        distances = torch.cdist(flat_x, self.embedding.weight)  # shape: (N * T, K)

        # get encoding
        min_distance, enc_indices = torch.min(distances, dim=-1)  # shape: (N * T, 1)
        fit_ = torch.mean(min_distance)

        # retrieve values based on the indices and reshape to N, C, T
        quantized = torch.index_select(self.embedding.weight, 0, enc_indices).view((N, width, T))

        # reshape x back to N, C, T
        x = x.permute(0, 2, 1).contiguous()
        commit_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + self.commit_cost * commit_loss

        quantized = x + (quantized - x).detach()

        return quantized, loss, fit_
