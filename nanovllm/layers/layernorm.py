import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Flag to disable compilation for training
        self._disable_compile = False

    def disable_compile(self):
        """Disable torch.compile for training with gradient computation."""
        self._disable_compile = True

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        if self._disable_compile:
            # Non-inplace version for training
            x = x * torch.rsqrt(var + self.eps)
            x = x.to(orig_dtype) * self.weight
        else:
            x.mul_(torch.rsqrt(var + self.eps))
            x = x.to(orig_dtype).mul_(self.weight)
        return x

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        if self._disable_compile:
            # Non-inplace version for training
            x = x * torch.rsqrt(var + self.eps)
            x = x.to(orig_dtype) * self.weight
        else:
            x.mul_(torch.rsqrt(var + self.eps))
            x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._disable_compile:
            # Use non-compiled version
            if residual is None:
                return self.rms_forward(x)
            else:
                return self.add_rms_forward(x, residual)
        else:
            # Use compiled version
            if residual is None:
                return self.rms_forward(x)
            else:
                return self.add_rms_forward(x, residual)
