from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_context: Optional[int] = None,
        n_heads: int = 8,
        dim_per_head: Optional[int] = None,
        dropout: float = 0.3,
        use_flash: bool = False
    ):
        super().__init__()
        self.use_flash = use_flash
        self.p_dropout = dropout

        if dim_context is None:
            dim_context = dim
        if dim_per_head is None:
            assert dim % n_heads == 0, f"dim must be divisible by no of attention heads. Got dim={dim}, n_heads={n_heads}"
            dim_per_head = dim // n_heads
        
        dim_inner = dim_per_head * n_heads
        self.scale = dim_per_head ** -0.5
        self.n_heads = n_heads
        self.dim_per_head = dim_per_head

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.out_proj = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (Tensor): Shape (B, Lq, D)
            context (optional Tensor): Shape (B, Lk, Dc)
            context_mask (optional Tensor): Boolean or float mask of context. If boolean, False indicates the element should be ignored. Shape (B, Lk) or (B, Lq, Lk)
        """
        batch_size, n_heads, dim_per_head = x.size(0), self.n_heads, self.dim_per_head
        if context is None:
            context = x

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        # (B, S, D) -> (B, H, S, d)
        q, k, v = [
            tensor.view(batch_size, -1, n_heads, dim_per_head)
            .transpose(1, 2).contiguous()
            for tensor in (q, k, v)
        ]

        if context_mask is not None:
            # Expand (B, Lk) to the shape (B, Lq, Lk)
            if context_mask.ndim == 2:
                context_mask = context_mask.unsqueeze(1).repeat_interleave(q.size(2))
            # Expand (B, Lq, Lk) to the shape (B, H, Lq, Lk)
            if context_mask.ndim == 3:
                context_mask = context_mask.unsqueeze(1).repeat_interleave(n_heads)

        with torch.backends.cuda.sdp_kernel(enable_flash=self.use_flash):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=context_mask,
                dropout_p=self.p_dropout if self.training else 0.0,
                scale=self.scale
            )
            # score = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
            # import pdb; pdb.set_trace()
        
        # (B, H, S, d) -> (B, S, D)
        out = out.transpose(1, 2).contiguous().view_as(x)
        return out

class FiLMLayer(nn.Conv2d):
    def __init__(
        self, in_channels: int, cond_channels: int, w_init_gain: Optional[str] = None
    ):
        self.w_init_gain = w_init_gain

        super().__init__(
            in_channels=cond_channels, out_channels=in_channels * 2, kernel_size=1
        )

    def reset_parameters(self):
        print("RESET FiLM params")
        if self.w_init_gain is None:
            super().reset_parameters()
        elif self.w_init_gain in ["zero"]:
            torch.nn.init.zeros_(self.weight)
        elif self.w_init_gain in ["relu", "leaky_relu"]:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity=self.w_init_gain)
        elif self.w_init_gain == "glu":
            assert (
                self.out_channels % 2 == 0
            ), "The out_channels of GLU requires even number."
            torch.nn.init.kaiming_uniform_(
                self.weight[: self.out_channels // 2], nonlinearity="linear"
            )
            torch.nn.init.xavier_uniform_(
                self.weight[self.out_channels // 2 :],
                gain=torch.nn.init.calculate_gain("sigmoid"),
            )
        elif self.w_init_gain == "gate":
            assert (
                self.out_channels % 2 == 0
            ), "The out_channels of GLU requires even number."
            torch.nn.init.xavier_uniform_(
                self.weight[: self.out_channels // 2],
                gain=torch.nn.init.calculate_gain("tanh"),
            )
            torch.nn.init.xavier_uniform_(
                self.weight[self.out_channels // 2 :],
                gain=torch.nn.init.calculate_gain("sigmoid"),
            )
        else:
            torch.nn.init.xavier_uniform_(
                self.weight, gain=torch.nn.init.calculate_gain(self.w_init_gain)
            )
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ):
        # cond.shape = (B, C, H, W)
        if cond_mask is not None:
            cond = cond * cond_mask
        # betas, gammas shape = (B, x_C, S)
        betas, gammas = super().forward(cond).chunk(2, 1)
        # betas, gammas = (t for t in (betas, gammas))
        x = gammas * x + betas

        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, f"Dim must be divisible by 2. Got {dim}"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def GroupNorm(in_channels: int, num_groups: int = 32):
    return nn.GroupNorm(num_groups, in_channels)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv2d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv2d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        # self.saln = StyleAdaptiveLayerNorm(d_in, style_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # output = x.transpose(1, 2)
        output = x
        w1 = self.w_1(output)
        relu1 = F.relu(w1)
        output = self.w_2(relu1)
        # output = output.transpose(1, 2)
        output = self.dropout(output)

        return output

class DenoiseEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dim_cond_mult: int,
        num_attn_heads: int,
        fft_conv1d_input_size,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        depth: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        dim_time = hidden_dim * dim_cond_mult
        num_attn_heads = num_attn_heads
        depth = depth
        dropout = dropout

        self.conv_in = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, input_dim, kernel_size=3, padding=1)
        self.conv_act = nn.SiLU()
        self.to_time_cond = nn.Sequential(
            LearnedSinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim + 1, dim_time),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GroupNorm(hidden_dim),
                        FiLMLayer(hidden_dim, dim_time, "linear"),
                        MultiHeadAttention(
                            hidden_dim, n_heads=num_attn_heads
                        ),
                        GroupNorm(hidden_dim),
                        FiLMLayer(hidden_dim, dim_time, "linear"),
                        PositionwiseFeedForward(
                            hidden_dim,
                            fft_conv1d_input_size,
                            fft_conv1d_kernel,
                            fft_conv1d_padding,
                        ),
                        nn.Dropout(dropout),
                    ]
                )
            )

        self.film = FiLMLayer(hidden_dim, dim_time, "linear")
        self.final_norm = GroupNorm(hidden_dim)
    
    def attn_fw(self, attn, x):
        b, c, *spatial = x.shape
        x = x.view(b, c, -1).transpose(-1, -2)
        x = attn(x)
        x = x.transpose(-1, -2).view(b, c, *spatial)
        return x

    def forward(self, x: torch.Tensor, t: int):
        # x.shape (B, C, H, W)
        x = self.conv_in(x)

        # t_embed.shape = (B, D)
        t_embed = self.to_time_cond(t)[:, :, None, None]
        for attn_norm, film1, attn, ff_norm, film2, ff, dropout in self.layers:
            x = self.attn_fw(attn, film1(attn_norm(x), t_embed)) + x
            x = ff(film2(ff_norm(x), t_embed)) + x

        x = self.film(self.final_norm(x), t_embed)
        
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x
