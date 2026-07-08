# model.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: [B] (long or float)
        return:    [B, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class GaussianFourierProjection(nn.Module):
    """Implementation details for GaussianFourierProjection."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        """Implementation details for __init__."""
        super().__init__()
        assert embed_dim % 2 == 0, 'Invalid input or configuration.'

        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation details for forward."""
        if x.dim() == 1:
            x = x[:, None]
        # x: [B, 1]
        # W: [F]
        x_proj = x[:, 0:1] * self.W[None, :] * 2 * np.pi  # [B, F]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B, 2F]


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)


        if out_channels < groups:
            groups = 1
        if out_channels % groups != 0:
            g = min(groups, out_channels)
            while out_channels % g != 0 and g > 1:
                g //= 2
            if g <= 0:
                g = 1
            groups = g

        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        # time embedding -> scale, shift
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2),
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = ConvBlock3D(in_channels, out_channels, groups=groups)
        self.block2 = ConvBlock3D(out_channels, out_channels, groups=groups, dropout=dropout)

        self.res_conv = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

        self.out_channels = out_channels

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            temb = self.mlp(time_emb)                 # [B, 2*C]
            temb = rearrange(temb, "b c -> b c 1 1 1")
            scale_shift = temb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h, scale_shift=scale_shift)
        return h + self.res_conv(x)


class Attention3D(nn.Module):
    """Implementation details for Attention3D."""

    def __init__(
        self,
        dim,
        heads: int = 4,
        dim_head: int = 32,
        use_rope: bool = True,
        rope_theta: float = 100.0,
        rope_z_factor: float = 1.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim


        self.use_rope = use_rope
        self.rope_theta_xy = float(rope_theta)

        self.rope_theta_z = float(rope_theta * rope_z_factor)
        self._rope_cache_key = None
        self._rope_cache = {}


        self.norm = nn.GroupNorm(1, dim)


        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, kernel_size=1),
            nn.GroupNorm(1, dim),
        )

    def _get_rope_cache(self, d: int, h: int, w: int, device: torch.device, dtype: torch.dtype, axis_dim: int):
        key = (d, h, w, device, dtype, axis_dim, self.rope_theta_xy, self.rope_theta_z)
        if self._rope_cache_key == key:
            return self._rope_cache

        pos_dtype = dtype
        x_pos = torch.arange(d, device=device, dtype=pos_dtype)
        y_pos = torch.arange(h, device=device, dtype=pos_dtype)
        z_pos = torch.arange(w, device=device, dtype=pos_dtype)
        xx, yy, zz = torch.meshgrid(x_pos, y_pos, z_pos, indexing="ij")
        xx = xx.reshape(1, 1, -1)
        yy = yy.reshape(1, 1, -1)
        zz = zz.reshape(1, 1, -1)

        half = axis_dim // 2
        idx = torch.arange(half, device=device, dtype=pos_dtype)
        inv_freq_xy = 1.0 / (self.rope_theta_xy ** (2 * idx / axis_dim))
        inv_freq_z = 1.0 / (self.rope_theta_z ** (2 * idx / axis_dim))

        angle_x = xx[..., None] * inv_freq_xy
        angle_y = yy[..., None] * inv_freq_xy
        angle_z = zz[..., None] * inv_freq_z

        sin_x, cos_x = angle_x.sin(), angle_x.cos()
        sin_y, cos_y = angle_y.sin(), angle_y.cos()
        sin_z, cos_z = angle_z.sin(), angle_z.cos()

        self._rope_cache = {
            "sin_x": sin_x,
            "cos_x": cos_x,
            "sin_y": sin_y,
            "cos_y": cos_y,
            "sin_z": sin_z,
            "cos_z": cos_z,
        }
        self._rope_cache_key = key
        return self._rope_cache

    def _apply_3d_rope(self, q: torch.Tensor, k: torch.Tensor, d: int, h: int, w: int):
        """Implementation details for _apply_3d_rope."""
        if not self.use_rope:
            return q, k

        b, n_heads, n, dim_head = q.shape
        device = q.device


        rotary_dim = dim_head
        rotary_dim = rotary_dim - (rotary_dim % 6)
        if rotary_dim < 6:
            return q, k

        axis_dim = rotary_dim // 3

        if axis_dim % 2 != 0:
            axis_dim -= 1
            rotary_dim = axis_dim * 3
        if axis_dim <= 0:
            return q, k


        q_rotary, q_rest = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rotary, k_rest = k[..., :rotary_dim], k[..., rotary_dim:]

        qx, qy, qz = torch.split(q_rotary, axis_dim, dim=-1)
        kx, ky, kz = torch.split(k_rotary, axis_dim, dim=-1)

        cache = self._get_rope_cache(d, h, w, device, q.dtype, axis_dim)
        sin_x, cos_x = cache["sin_x"], cache["cos_x"]
        sin_y, cos_y = cache["sin_y"], cache["cos_y"]
        sin_z, cos_z = cache["sin_z"], cache["cos_z"]

        def apply_rope_axis_cached(q_axis, k_axis, sin, cos):
            """
            q_axis, k_axis: [B, H, N, axis_dim]
            sin, cos:       [1, 1, N, half]
            """
            half = axis_dim // 2
            q_even, q_odd = q_axis[..., :half], q_axis[..., half:]
            k_even, k_odd = k_axis[..., :half], k_axis[..., half:]

            q_axis_rot = torch.cat(
                [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos],
                dim=-1,
            )
            k_axis_rot = torch.cat(
                [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos],
                dim=-1,
            )
            return q_axis_rot, k_axis_rot


        qx, kx = apply_rope_axis_cached(qx, kx, sin_x, cos_x)
        qy, ky = apply_rope_axis_cached(qy, ky, sin_y, cos_y)
        qz, kz = apply_rope_axis_cached(qz, kz, sin_z, cos_z)

        q_rotary = torch.cat([qx, qy, qz], dim=-1)
        k_rotary = torch.cat([kx, ky, kz], dim=-1)

        q = torch.cat([q_rotary, q_rest], dim=-1)
        k = torch.cat([k_rotary, k_rest], dim=-1)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        """
        b, c, d, h, w = x.shape
        residual = x


        x = self.norm(x)                              # [B, C, D, H, W]
        x = rearrange(x, "b c d h w -> b c (d h w)")  # [B, C, N]


        qkv = self.to_qkv(x).chunk(3, dim=1)          # 3 x [B, inner_dim, N]

        def reshape_to_heads(t):
            # [B, inner_dim, N] -> [B, heads, N, dim_head]
            return rearrange(t, "b (h c) n -> b h n c", h=self.heads)

        q, k, v = map(reshape_to_heads, qkv)          # [B, H, N, Dh]

        # 3D Rotary Positional Embeddings
        q, k = self._apply_3d_rope(q, k, d, h, w)


        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )  # [B, H, N, Dh]

        out = rearrange(attn_out, "b h n c -> b (h c) n")  # [B, inner_dim, N]
        out = self.to_out(out)                             # [B, C, N]
        out = rearrange(out, "b c (d h w) -> b c d h w", d=d, h=h, w=w)

        return residual + out


class Downsample3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------------------------------------------------------
# 3. Rectified Flow Matching UNet3D
# -------------------------------------------------------------------------
class UNet3DRectifiedFlow(nn.Module):
    """Implementation details for UNet3DRectifiedFlow."""

    def __init__(
        self,
        dim: int,
        dim_mults=(1, 2, 4),
        in_channels: int = 32,
        out_channels: int = 32,
        cond_channels: int = 0,
        dropout: float = 0.0,
        groups: int = 8,
        use_coord: bool = True,
        coord_channels: int = 3,
        use_mid_attn: bool = True,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        attn_down_levels=(1, 2),
        rope_theta: float = 10000.0,
        rope_z_factor: float = 2.0,
        use_scale_emb: bool = True,
        scale_fourier_dim: int = 64,
        use_log_scale: bool = False,
        log_scale_eps: float = 1e-6,
        scale_unknown_value: float = -1.0,
        use_depth_fourier: bool = True,
        depth_fourier_dim: int | None = None,
        depth_unknown_value: float = -1.0,
    ):
        super().__init__()


        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.use_coord = use_coord
        self.coord_channels = coord_channels if use_coord else 0
        self.use_scale_emb = use_scale_emb
        self.use_log_scale = bool(use_log_scale)
        self.log_scale_eps = float(log_scale_eps)
        self.scale_unknown_value = float(scale_unknown_value)
        self.rope_theta = float(rope_theta)
        self.rope_z_factor = float(rope_z_factor)
        self.use_depth_fourier = bool(use_depth_fourier)
        self.depth_unknown_value = float(depth_unknown_value)

        extra_in = cond_channels + self.coord_channels
        total_in_channels = in_channels + extra_in


        dims = [total_in_channels, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)


        self.down_channels = [dim_out for (_, dim_out) in in_out]



        time_base_dim = dim
        self.time_embed = SinusoidalPositionEmbeddings(time_base_dim)


        time_mlp_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_base_dim, time_mlp_dim),
            nn.GELU(),
            nn.Linear(time_mlp_dim, time_mlp_dim),
        )
        self.time_emb_dim = time_mlp_dim




        if self.use_scale_emb:
            assert scale_fourier_dim % 2 == 0, 'Invalid input or configuration.'
            self.scale_fourier = GaussianFourierProjection(embed_dim=scale_fourier_dim)
            self.unknown_scale_fourier = nn.Parameter(torch.zeros(1, scale_fourier_dim))
            if self.use_depth_fourier:
                if depth_fourier_dim is None:
                    depth_fourier_dim = scale_fourier_dim
                assert depth_fourier_dim % 2 == 0, 'Invalid input or configuration.'
                self.depth_fourier = GaussianFourierProjection(embed_dim=depth_fourier_dim)
                self.unknown_depth_fourier = nn.Parameter(torch.zeros(1, depth_fourier_dim))
                depth_dim = depth_fourier_dim
            else:
                self.depth_fourier = None
                self.unknown_depth_fourier = None
                depth_dim = 1

            self.scale_depth_mlp = nn.Sequential(
                nn.Linear(scale_fourier_dim + depth_dim, time_mlp_dim),
                nn.SiLU(),
                nn.Linear(time_mlp_dim, time_mlp_dim),
            )
        else:
            self.scale_fourier = None
            self.scale_depth_mlp = None
            self.unknown_scale_fourier = None
            self.depth_fourier = None
            self.unknown_depth_fourier = None

        # ------------------- Encoder (Down) -------------------
        self.downs = nn.ModuleList([])
        self.attn_downs = nn.ModuleList([])
        attn_down_levels = set(attn_down_levels or ())

        for level, (dim_in, dim_out) in enumerate(in_out):
            is_last = level == num_resolutions - 1

            block1 = ResnetBlock3D(
                dim_in,
                dim_out,
                time_emb_dim=self.time_emb_dim,
                dropout=dropout,
                groups=groups,
            )
            block2 = ResnetBlock3D(
                dim_out,
                dim_out,
                time_emb_dim=self.time_emb_dim,
                dropout=dropout,
                groups=groups,
            )
            downsample = Downsample3D(dim_out) if not is_last else nn.Identity()

            self.downs.append(nn.ModuleList([block1, block2, downsample]))


            if level in attn_down_levels:
                self.attn_downs.append(
                    Attention3D(
                        dim_out,
                        heads=attn_heads,
                        dim_head=attn_dim_head,
                        rope_theta=self.rope_theta,
                        rope_z_factor=self.rope_z_factor,
                    )
                )
            else:
                self.attn_downs.append(nn.Identity())

        # ------------------- Bottleneck -------------------
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock3D(
            mid_dim, mid_dim,
            time_emb_dim=self.time_emb_dim,
            dropout=dropout,
            groups=groups,
        )
        self.mid_attn = (
            Attention3D(
                mid_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                rope_theta=self.rope_theta,
                rope_z_factor=self.rope_z_factor,
            )
            if use_mid_attn else nn.Identity()
        )
        self.mid_block2 = ResnetBlock3D(
            mid_dim, mid_dim,
            time_emb_dim=self.time_emb_dim,
            dropout=dropout,
            groups=groups,
        )

        # ------------------- Decoder (Up) -------------------
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == num_resolutions - 1

            block1 = ResnetBlock3D(
                dim_out * 2,
                dim_out,
                time_emb_dim=self.time_emb_dim,
                dropout=dropout,
                groups=groups,
            )
            block2 = ResnetBlock3D(
                dim_out,
                dim_out,
                time_emb_dim=self.time_emb_dim,
                dropout=dropout,
                groups=groups,
            )
            upsample = Upsample3D(dim_out, dim_in) if not is_last else nn.Identity()

            self.ups.append(nn.ModuleList([block1, block2, upsample]))


        self.final_conv = nn.Conv3d(dims[1], out_channels, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)


    def _build_coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation details for _build_coord_grid."""
        if not self.use_coord:
            return None

        b, _, d, h, w = x.shape
        device = x.device
        dtype = x.dtype

        x_lin = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
        y_lin = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        z_lin = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        xx, yy, zz = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=0)      # [3, D, H, W]
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)
        return grid


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        cond: torch.Tensor = None,
        phys_scale: torch.Tensor = None,
        coords_norm: torch.Tensor = None,
        control_down=None,
        control_mid=None,
        control_up=None,
        control_feats=None,
    ):
        """Implementation details for forward."""

        if control_feats is not None and control_down is None:
            control_down = control_feats


        if self.cond_channels > 0:
            assert cond is not None, 'Invalid input or configuration.'
            x = torch.cat([x, cond], dim=1)


        if self.use_coord:
            coord_grid = self._build_coord_grid(x)
            x = torch.cat([x, coord_grid], dim=1)



        t_emb = self.time_embed(t)          # [B, time_base_dim]
        t_emb = self.time_mlp(t_emb)        # [B, time_emb_dim]


        if self.use_scale_emb and (phys_scale is not None):

            if phys_scale.dim() == 1:
                s_scalar = phys_scale[:, None]
            elif phys_scale.dim() == 2:
                if phys_scale.size(1) == 1:
                    s_scalar = phys_scale
                else:

                    s_scalar = phys_scale.norm(dim=1, keepdim=True)
            else:
                raise ValueError(f'Invalid input or configuration.')

            unknown_mask = (s_scalar == self.scale_unknown_value)
            if self.use_log_scale:
                s_safe = torch.clamp(s_scalar, min=self.log_scale_eps)
                s_scalar = torch.where(unknown_mask, s_scalar, torch.log(s_safe))



            if coords_norm is None:
                z_scalar = torch.full_like(s_scalar, self.depth_unknown_value)
            elif coords_norm.dim() == 1:
                z_scalar = coords_norm[:, None]
            else:
                z_scalar = coords_norm[:, :1]


            s_fourier = self.scale_fourier(s_scalar)   # [B, F]
            if self.unknown_scale_fourier is not None:
                mask_f = unknown_mask.to(s_fourier.dtype)
                s_fourier = s_fourier * (1.0 - mask_f) + self.unknown_scale_fourier * mask_f

            if self.use_depth_fourier and (self.depth_fourier is not None):
                z_fourier = self.depth_fourier(z_scalar)
                if self.unknown_depth_fourier is not None:
                    depth_unknown_mask = (z_scalar == self.depth_unknown_value)
                    mask_f = depth_unknown_mask.to(z_fourier.dtype)
                    z_fourier = z_fourier * (1.0 - mask_f) + self.unknown_depth_fourier * mask_f
                z_feat = z_fourier
            else:
                z_feat = z_scalar

            sd_input = torch.cat([s_fourier, z_feat], dim=-1)

            sd_emb = self.scale_depth_mlp(sd_input)            # [B, time_emb_dim]

            time_embed = t_emb + sd_emb
        else:
            time_embed = t_emb

        hs = []

        # ------------------- Down path -------------------
        for level, ((block1, block2, downsample), attn_block) in enumerate(
            zip(self.downs, self.attn_downs)
        ):
            x = block1(x, time_embed)


            if control_down is not None and level < len(control_down) and control_down[level] is not None:
                x = x + control_down[level]

            x = block2(x, time_embed)
            x = attn_block(x)

            hs.append(x)
            x = downsample(x)

        # ------------------- Middle -------------------
        x = self.mid_block1(x, time_embed)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_embed)


        if control_mid is not None:
            x = x + control_mid

        # ------------------- Up path -------------------
        for level, (block1, block2, upsample) in enumerate(self.ups):
            residual = hs.pop()
            x = torch.cat([x, residual], dim=1)

            x = block1(x, time_embed)
            x = block2(x, time_embed)


            if control_up is not None and level < len(control_up) and control_up[level] is not None:
                x = x + control_up[level]

            x = upsample(x)


        v = self.final_conv(x)
        return v



GeoRectifiedFlowUNet3D = UNet3DRectifiedFlow


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
class GeoControlNet3D(nn.Module):
    """Implementation details for GeoControlNet3D."""

    def __init__(
            self,
            cond_in_channels: int,
            unet: nn.Module,
            full_res: int = 128,
            latent_res: int = 16,
            groups: int = 8,
            time_emb_dim: int = None,
    ):
        super().__init__()
        assert full_res % latent_res == 0, 'Invalid input or configuration.'
        self.full_res = full_res
        self.latent_res = latent_res


        self.down_channels = list(unet.down_channels)


        first_ch = self.down_channels[0]
        pre_blocks = []
        in_ch = cond_in_channels
        out_ch = first_ch
        cur_res = full_res

        while cur_res > latent_res:
            pre_blocks.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            g = groups
            if out_ch < g or out_ch % g != 0: g = 1
            pre_blocks.append(nn.GroupNorm(g, out_ch))
            pre_blocks.append(nn.SiLU())
            in_ch = out_ch
            cur_res //= 2

        self.pre = nn.Sequential(*pre_blocks)


        self.control_down_blocks = nn.ModuleList()
        self.control_downsamples = nn.ModuleList()

        prev_ch = first_ch
        num_levels = len(self.down_channels)

        for level, ch_out in enumerate(self.down_channels):



            block = nn.Sequential(
                ResnetBlock3D(prev_ch, ch_out, time_emb_dim=time_emb_dim, groups=groups),
                ResnetBlock3D(ch_out, ch_out, time_emb_dim=time_emb_dim, groups=groups),
            )
            self.control_down_blocks.append(block)
            prev_ch = ch_out


            if level < num_levels - 1:
                self.control_downsamples.append(Downsample3D(ch_out))

    def forward(self, cond_vox: torch.Tensor, emb: torch.Tensor = None):
        """Implementation details for forward."""
        x = self.pre(cond_vox)
        feats = []

        for level, block in enumerate(self.control_down_blocks):

            for layer in block:
                if isinstance(layer, ResnetBlock3D):

                    x = layer(x, time_emb=emb)
                else:
                    x = layer(x)

            feats.append(x)

            if level < len(self.control_downsamples):
                x = self.control_downsamples[level](x)

        return feats
