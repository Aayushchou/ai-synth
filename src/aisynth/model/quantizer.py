import torch.nn as nn
import torch
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
     Main VQVAE constriction block. Creates a codebook and quantises embeddings by selecting vectors from the
     codebook that have the lowest cosine distance to the embedding. The codebook vectors are also trained.
     Encoder - VectorQuantizer - Decoder

     params:
        :emb_width - width of the embeddings passed from the encoder
        :n_bins - number of vectors we want to have within the codebook
        :commit_cost - parameter to provide custom weighting to the commit loss calculation

    attrs:
        :self.embedding -- contains the codebook vectors
    """

    def __init__(self, emb_width: int, n_bins: int, commit_cost: float):
        super().__init__()
        self.emb_width = emb_width
        self.n_bins = n_bins
        self.commit_cost = commit_cost
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.embedding = nn.Embedding(self.n_bins, self.emb_width)
        self.embedding.weight.data.uniform_(-1 / self.n_bins, 1 / self.n_bins)

    def forward(self, x: torch.Tensor):
        """
        performs vector quantization given an input embedding from the encoder.

        procedure:
            1. reshape input to (num_samples, timesteps, channels)
            2. flatten to (num_samples * timesteps, channels)
            3. calculate cosine distances to the embedding weights (num_samples * timesteps, n_bins)
            4. get indices of the closest vectors in the embedding space (num_samples * timesteps, 1)
            5. Retrieve embeddings using the indices from step 4 (num_samples, channels, timesteps)
            6. Reshape input back to (num_samples, channels, timesteps)
            7. Compute codebook loss and commit loss
            8. Gradient passthrough

        :param x: input embeddings outputted from the encoder
        :return: quantised representation of the input embeddings
        """
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
        quantized = torch.index_select(self.embedding.weight, 0, enc_indices).view(
            (N, width, T)
        )

        # reshape x back to N, C, T
        x = x.permute(0, 2, 1).contiguous()

        # compute loss
        commit_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + self.commit_cost * commit_loss

        # gradient pass-through
        quantized = x + (quantized - x).detach()

        return quantized, loss, fit_
