from __future__ import annotations

"""Optional VQ implementations used by the IDM backbone."""

import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
import torch.nn as nn
import torch.nn.functional as F


class SimpleNSVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        discarding_threshold: float = 0.01,
        initialization: str = 'normal',
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.discarding_threshold = discarding_threshold
        self.eps = eps
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        self.register_buffer('is_initialized', torch.tensor(0, dtype=torch.uint8))
        self.register_buffer('codebook_usage', torch.zeros(codebook_size, dtype=torch.long))

    def _init_codebook(self, x_flat: torch.Tensor) -> None:
        with torch.no_grad():
            n_data = x_flat.size(0)
            if n_data < self.codebook_size:
                indices = torch.randint(0, n_data, (self.codebook_size,), device=x_flat.device)
            else:
                indices = torch.randperm(n_data, device=x_flat.device)[: self.codebook_size]
            self.codebook.data.copy_(x_flat[indices].clone())
            self.is_initialized.fill_(1)

    def forward(self, x: torch.Tensor, codebook_training_only: bool = False):
        x_flat = x.reshape(-1, self.dim)
        if self.training and self.is_initialized.item() == 0:
            self._init_codebook(x_flat)

        x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook ** 2).sum(dim=1)
        distances = x_sq - 2 * (x_flat @ self.codebook.t()) + e_sq.unsqueeze(0)
        indices = torch.argmin(distances, dim=1)
        codes = self.codebook[indices]

        resid = x_flat - codes
        resid_norm = resid.norm(dim=1, keepdim=True)
        noise = torch.randn_like(x_flat)
        noise_norm = noise.norm(dim=1, keepdim=True)
        scaled_noise = (resid_norm / (noise_norm + self.eps)) * noise
        quantized_flat = codes if codebook_training_only else x_flat + scaled_noise

        commitment_loss = F.mse_loss(x_flat, codes.detach())
        codebook_loss = F.mse_loss(x_flat.detach(), codes)
        vq_loss = codebook_loss + 0.25 * commitment_loss

        if self.training:
            with torch.no_grad():
                self.codebook_usage.index_add_(0, indices, torch.ones_like(indices, dtype=torch.long))

        quantized = quantized_flat.view(*x.shape)
        indices = indices.view(*x.shape[:-1])
        return quantized, indices, vq_loss

    @torch.no_grad()
    def replace_unused_codebooks(self, num_batches: int) -> None:
        if num_batches <= 0:
            return
        usage_rate = self.codebook_usage.float() / float(num_batches)
        unused = torch.where(usage_rate < self.discarding_threshold)[0]
        used = torch.where(usage_rate >= self.discarding_threshold)[0]

        if used.numel() == 0:
            self.codebook.add_(self.eps * torch.randn_like(self.codebook))
        elif unused.numel() > 0:
            used_codes = self.codebook[used]
            idx = torch.randint(0, used_codes.size(0), (unused.size(0),), device=used_codes.device)
            self.codebook[unused] = used_codes[idx] + torch.randn_like(self.codebook[unused]) * 0.02
        self.codebook_usage.zero_()


class NSVQ(nn.Module):
    def __init__(
        self,
        dim,
        num_embeddings,
        embedding_dim,
        device=torch.device('cpu'),
        discarding_threshold=0.1,
        initialization='normal',
        code_seq_len=1,
        patch_size=32,
        image_size=256,
        is_vector_input=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim
        self.patch_size = patch_size
        self.is_vector_input = is_vector_input

        if initialization == 'normal':
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(
                -1 / self.num_embeddings, 1 / self.num_embeddings
            ).sample([self.num_embeddings, self.embedding_dim])
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")
        self.codebooks = nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)
        self.project_in = nn.Linear(dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, dim)

        if self.is_vector_input:
            self.cnn_encoder = nn.Identity()
        elif code_seq_len == 1:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=1, padding=0),
            )
        elif code_seq_len == 2:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(3, 4), stride=1, padding=0),
            )
        elif code_seq_len == 4:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=0),
            )
        elif code_seq_len == 16:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            )
        elif code_seq_len == 64:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            )
        elif code_seq_len == 256:
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError('Not implemented: code_seq_len should be one of {1,2,4,16,64,256}')

    def encode(self, input_data: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.is_vector_input:
            return self.project_in(input_data)
        input_data = self.project_in(input_data)
        input_data = input_data.permute(0, 2, 1).contiguous()
        input_data = input_data.reshape(
            batch_size,
            self.embedding_dim,
            int(self.image_size / self.patch_size),
            int(self.image_size / self.patch_size),
        )
        input_data = self.cnn_encoder(input_data)
        input_data = input_data.reshape(batch_size, self.embedding_dim, -1)
        input_data = input_data.permute(0, 2, 1).contiguous()
        return input_data.reshape(-1, self.embedding_dim)

    def decode(self, quantized_input: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.is_vector_input:
            return self.project_out(quantized_input)
        quantized_input = quantized_input.reshape(batch_size, self.embedding_dim, -1)
        quantized_input = quantized_input.permute(0, 2, 1).contiguous()
        return self.project_out(quantized_input)

    def forward(self, input_data_first: torch.Tensor, input_data_last: torch.Tensor, codebook_training_only: bool = False):
        batch_size = input_data_first.shape[0]
        input_data_first = self.encode(input_data_first.contiguous(), batch_size)
        input_data_last = self.encode(input_data_last, batch_size)
        input_data = input_data_last - input_data_first

        distances = (
            torch.sum(input_data ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(input_data, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        random_vector = normal_dist.Normal(0, 1).sample(input_data.shape).to(self.device)
        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = (norm_quantization_residual / norm_random_vector + self.eps) * random_vector
        quantized_input = hard_quantized_input if codebook_training_only else input_data + vq_error

        encodings = torch.zeros(input_data.shape[0], self.num_embeddings, device=input_data.device)
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        quantized_input = self.decode(quantized_input, batch_size)
        return quantized_input, perplexity, self.codebooks_used.cpu().numpy(), min_indices.reshape(batch_size, -1)

    @torch.no_grad()
    def replace_unused_codebooks(self, num_batches):
        unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
        used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]
        unused_count = unused_indices.shape[0]
        used_count = used_indices.shape[0]
        if used_count == 0:
            self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
        else:
            used = self.codebooks[used_indices].clone()
            if used_count < unused_count:
                used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
            else:
                used_codebooks = used
            self.codebooks[unused_indices] *= 0
            self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                (unused_count, self.embedding_dim), device=self.device
            ).clone()
        self.codebooks_used[:] = 0.0

    def inference(self, input_data_first, input_data_last, user_action_token_num=None):
        input_data_first = input_data_first.detach().clone()
        input_data_last = input_data_last.detach().clone()
        codebooks = self.codebooks.detach().clone()
        batch_size = input_data_first.shape[0]
        input_data_first = self.encode(input_data_first, batch_size)
        input_data_last = self.encode(input_data_last, batch_size)
        input_data = (input_data_last - input_data_first).reshape(-1, self.embedding_dim)
        distances = (
            torch.sum(input_data ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(input_data, codebooks.t())
            + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True)
        )
        min_indices = torch.argmin(distances, dim=1)
        if user_action_token_num is not None:
            if isinstance(user_action_token_num, list):
                min_indices = torch.tensor(user_action_token_num, device=self.device)
            else:
                min_indices = torch.tensor([[user_action_token_num]], device=self.device).repeat(input_data.shape[0], 1)
        quantized_input = codebooks[min_indices]
        quantized_input = self.decode(quantized_input, batch_size)
        return quantized_input, min_indices.reshape(batch_size, -1)

    def codebook_reinit(self):
        self.codebooks = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim, device=self.device), requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=self.device)
