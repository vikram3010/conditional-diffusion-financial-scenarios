import torch


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, num_steps)


class DiffusionSchedule:
    def __init__(self, num_steps: int = 300, device: str = "cpu"):
        self.num_steps = num_steps
        self.device = device

        self.betas = linear_beta_schedule(num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(0, self.num_steps, (batch_size,), device=self.device)

    def add_noise(self, clean_returns: torch.Tensor, timesteps: torch.Tensor):
        noise = torch.randn_like(clean_returns)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timesteps]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars[timesteps]).view(-1, 1, 1)

        noisy_returns = sqrt_alpha_bar * clean_returns + sqrt_one_minus_alpha_bar * noise
        return noisy_returns, noise
