import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))

        return embedding


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, dilation: int):
        super().__init__()

        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.time_projection = nn.Linear(time_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        time_context = self.time_projection(time_embedding).unsqueeze(-1)
        h = h + time_context

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return residual + h


class ConditionalDDPM(nn.Module):
    def __init__(
        self,
        return_dim: int = 3,
        macro_dim: int = 5,
        hidden_dim: int = 96,
        time_dim: int = 192,
        num_blocks: int = 8,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_projection = nn.Conv1d(
            return_dim + macro_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    hidden_dim=hidden_dim,
                    time_dim=time_dim,
                    dilation=2 ** (i % 4),
                )
                for i in range(num_blocks)
            ]
        )

        self.output_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, return_dim, kernel_size=1),
        )

    def forward(
        self,
        noisy_returns: torch.Tensor,
        macro_context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([noisy_returns, macro_context], dim=-1)
        x = x.transpose(1, 2)

        time_embedding = self.time_embedding(timesteps)

        h = self.input_projection(x)

        for block in self.blocks:
            h = block(h, time_embedding)

        predicted_noise = self.output_head(h)
        return predicted_noise.transpose(1, 2)
