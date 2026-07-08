# model.py
# Geo-C3RF - Structure-Aware Dual-Head 3D VAE
#






from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Sequence


# -----------------------------------------------------------------------------#
# Utils
# -----------------------------------------------------------------------------#
VALID_VAE_VARIANTS = ("dual_head", "semantic_only", "edge_input_no_head")


def resolve_vae_variant(vae_variant: str) -> Tuple[bool, bool]:
    """
    Return (use_edge_input, use_edge_head) for the requested ablation variant.
    """
    if vae_variant not in VALID_VAE_VARIANTS:
        raise ValueError(f"Unknown vae_variant='{vae_variant}', expected one of {VALID_VAE_VARIANTS}")
    if vae_variant == "dual_head":
        return True, True
    if vae_variant == "semantic_only":
        return False, False
    if vae_variant == "edge_input_no_head":
        return True, False
    raise AssertionError(f"Unhandled vae_variant='{vae_variant}'")


def GN(c: int, groups: int = 4) -> nn.GroupNorm:
    if c < groups:
        return nn.GroupNorm(num_groups=max(1, c), num_channels=c)
    elif c % groups == 0:
        return nn.GroupNorm(num_groups=groups, num_channels=c)
    else:
        g = groups
        while g > 1 and c % g != 0:
            g //= 2
        if c % g != 0:
            g = 1
        return nn.GroupNorm(num_groups=g, num_channels=c)


def init_weights(module: nn.Module, gain: float = 0.5):
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.GroupNorm, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


# -----------------------------------------------------------------------------#
# Building blocks
# -----------------------------------------------------------------------------#
class ResBlock3D(nn.Module):
    """
    Conv3d -> GN -> SiLU -> Dropout -> Conv3d -> GN with residual.
    """

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: Optional[int] = None,
                 groups: int = 4, dropout_prob: float = 0.0):
        super().__init__()
        h = hidden_ch or out_ch
        self.conv1 = nn.Conv3d(in_ch, h, kernel_size=3, padding=1)
        self.gn1 = GN(h, groups)
        self.act1 = nn.SiLU(inplace=False)
        self.dropout = nn.Dropout3d(dropout_prob)
        self.conv2 = nn.Conv3d(h, out_ch, kernel_size=3, padding=1)
        self.gn2 = GN(out_ch, groups)
        self.skip = (nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return F.silu(x + residual, inplace=True)


class DownBlock3D(nn.Module):
    """
    Downsample by stride-2 conv, then a ResBlock.
    D -> D/2
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 4, dropout_prob: float = 0.0):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.rb = ResBlock3D(out_ch, out_ch, groups=groups, dropout_prob=dropout_prob)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.rb(x)
        return x


class UpBlock3D(nn.Module):
    """Implementation details for UpBlock3D."""

    def __init__(self, in_ch: int, out_ch: int,
                 groups: int = 4, dropout_prob: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.rb = ResBlock3D(out_ch, out_ch, groups=groups, dropout_prob=dropout_prob)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.rb(x)
        return x


# -----------------------------------------------------------------------------#
# 3D Self-Attention
# -----------------------------------------------------------------------------#
class SelfAttention3D(nn.Module):
    """Implementation details for SelfAttention3D."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True  # (B, N, C)
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        N = D * H * W

        x_flat = x.view(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)         # (B, N, C)
        attn_out = self.proj_out(attn_out)

        out = x_flat + attn_out
        out = out.permute(0, 2, 1).contiguous().view(B, C, D, H, W)
        return out


# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
class Encoder3D_Struct(nn.Module):
    """Implementation details for Encoder3D_Struct."""

    def __init__(self, in_channels: int, z_channels: int = 32,
                 base_ch: int = 16, groups: int = 4, dropout_prob: float = 0.0):
        super().__init__()
        C_in = in_channels
        ch1, ch2, ch3, ch4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # stem @ D
        self.stem = nn.Sequential(
            nn.Conv3d(C_in, ch1, kernel_size=3, padding=1),
            GN(ch1, groups), nn.SiLU(inplace=True),
            ResBlock3D(ch1, ch1, groups=groups, dropout_prob=dropout_prob)
        )

        # D -> D/2 -> D/4 -> D/8
        self.down1 = DownBlock3D(ch1, ch2, groups=groups, dropout_prob=dropout_prob)  # D -> D/2
        self.down2 = DownBlock3D(ch2, ch3, groups=groups, dropout_prob=dropout_prob)  # D/2 -> D/4
        self.down3 = DownBlock3D(ch3, ch4, groups=groups, dropout_prob=dropout_prob)  # D/4 -> D/8


        self.bottleneck = ResBlock3D(ch4, ch4, groups=groups, dropout_prob=dropout_prob)
        self.bottleneck_attn = SelfAttention3D(ch4, num_heads=4)

        # VAE heads
        self.conv_mu = nn.Conv3d(ch4, z_channels, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv3d(ch4, z_channels, kernel_size=3, padding=1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation details for forward."""
        fD = self.stem(x)           # [B, ch1, D,   D,   D]
        fD2 = self.down1(fD)        # [B, ch2, D/2, D/2, D/2]
        fD4 = self.down2(fD2)       # [B, ch3, D/4, D/4, D/4]
        fD8 = self.down3(fD4)       # [B, ch4, D/8, D/8, D/8]

        h = self.bottleneck(fD8)
        h = self.bottleneck_attn(h)  # attention @ D/8

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    def forward_with_feats(self, x: torch.Tensor):
        """Implementation details for forward_with_feats."""
        fD = self.stem(x)
        fD2 = self.down1(fD)
        fD4 = self.down2(fD2)
        fD8 = self.down3(fD4)

        h = self.bottleneck(fD8)
        h = self.bottleneck_attn(h)

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return (fD, fD2, fD4, h), (mu, logvar)


# -----------------------------------------------------------------------------#
# 2) Decoder Trunk + Dual Heads
# -----------------------------------------------------------------------------#
class Decoder3D_DualHead(nn.Module):
    """Implementation details for Decoder3D_DualHead."""

    def __init__(self, num_semantic_classes: int,
                 num_edge_channels: int = 3,
                 z_channels: int = 32,
                 base_ch: int = 16,
                 groups: int = 4,
                 dropout_prob: float = 0.0,
                 use_edge_head: bool = True):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.num_edge_channels = num_edge_channels
        self.use_edge_head = bool(use_edge_head)

        ch1, ch2, ch3, ch4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # map z -> ch4 @ D/8
        self.z_proj = nn.Sequential(
            nn.Conv3d(z_channels, ch4, kernel_size=3, padding=1),
            GN(ch4, groups), nn.SiLU(inplace=True),
            ResBlock3D(ch4, ch4, groups=groups, dropout_prob=dropout_prob),
        )

        # Up path: D/8(ch4) -> D/4(ch3) -> D/2(ch2) -> D(ch1)
        self.up1 = UpBlock3D(ch4, ch3, groups=groups, dropout_prob=dropout_prob)
        self.up2 = UpBlock3D(ch3, ch2, groups=groups, dropout_prob=dropout_prob)
        self.up3 = UpBlock3D(ch2, ch1, groups=groups, dropout_prob=dropout_prob)


        self.trunk_out = ResBlock3D(ch1, ch1, groups=groups, dropout_prob=dropout_prob)


        self.head_sem = nn.Conv3d(ch1, num_semantic_classes, kernel_size=3, padding=1)


        self.head_edge = (
            nn.Conv3d(ch1, num_edge_channels, kernel_size=3, padding=1)
            if self.use_edge_head else None
        )

        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, Z, D/8, D/8, D/8]
        """
        h = self.z_proj(z)   # [B, ch4, D/8, D/8, D/8]
        h = self.up1(h)      # [B, ch3, D/4, ...]
        h = self.up2(h)      # [B, ch2, D/2, ...]
        h = self.up3(h)      # [B, ch1, D,   D,   D]

        h = self.trunk_out(h)  # [B, ch1, D, D, D]

        logits_sem = self.head_sem(h)    # [B, C_sem, D, D, D]
        logits_edge = self.head_edge(h) if self.head_edge is not None else None  # [B, C_edge, D, D, D] or None

        return logits_sem, logits_edge


# -----------------------------------------------------------------------------#
# 3) Dual-Head VAE Wrapper
# -----------------------------------------------------------------------------#
class AutoencoderKL_3D_DualHead(nn.Module):
    """Implementation details for AutoencoderKL_3D_DualHead."""

    def __init__(self,
                 num_semantic_classes: int,
                 num_edge_channels: int = 3,
                 z_channels: int = 32,
                 input_shape: Tuple[int, int, int] = (128, 128, 128),
                 base_ch: int = 16,
                 groups: int = 4,
                 dropout_prob: float = 0.2,
                 vae_variant: str = "dual_head"):
        super().__init__()
        use_edge_input, use_edge_head = resolve_vae_variant(vae_variant)
        self.vae_variant = vae_variant
        self.use_edge_input = use_edge_input
        self.use_edge_head = use_edge_head
        self.num_semantic_classes = num_semantic_classes
        self.num_edge_channels = num_edge_channels
        self.in_channels = num_semantic_classes + (num_edge_channels if self.use_edge_input else 0)
        self.z_channels = z_channels
        self.input_shape = input_shape
        self.base_ch = base_ch
        self.groups = groups
        self.dropout_prob = dropout_prob

        print(f"--- 3D Dual-Head VAE (Struct-Aware) Initialized ---")
        print(f"Variant:          {self.vae_variant} | edge_input={self.use_edge_input} | edge_head={self.use_edge_head}")
        print(f"Input Channels:   {self.in_channels} = "
              f"{num_semantic_classes}(semantics)"
              f"{' + ' + str(num_edge_channels) + '(edges)' if self.use_edge_input else ''}")
        print(f"Base Channels:    {base_ch}, Latent Channels: {z_channels}, "
              f"Groups: {groups}, Dropout: {dropout_prob}")
        print(f"Input Shape:      (B, {self.in_channels}, "
              f"{input_shape[0]}, {input_shape[1]}, {input_shape[2]})")
        print(f"Latent Shape:     (B, {z_channels}, "
              f"{input_shape[0] // 8}, {input_shape[1] // 8}, {input_shape[2] // 8})")
        print(f"Head A (Sem):     C = {num_semantic_classes}")
        print(f"Head B (Edges):   C = {num_edge_channels if self.use_edge_head else 0}")

        self.encoder = Encoder3D_Struct(
            in_channels=self.in_channels,
            z_channels=z_channels,
            base_ch=base_ch,
            groups=groups,
            dropout_prob=dropout_prob,
        )

        self.decoder = Decoder3D_DualHead(
            num_semantic_classes=num_semantic_classes,
            num_edge_channels=num_edge_channels,
            z_channels=z_channels,
            base_ch=base_ch,
            groups=groups,
            dropout_prob=dropout_prob,
            use_edge_head=self.use_edge_head,
        )

    # ---------------------- VAE core ---------------------- #
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation details for encode."""
        mu, logvar = self.encoder(x)
        return mu, logvar

    def _encode_with_skips(self, x: torch.Tensor):
        """Implementation details for _encode_with_skips."""
        return self.encoder.forward_with_feats(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation details for decode."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """Implementation details for forward."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits_sem, logits_edge = self.decode(z)
        return logits_sem, logits_edge, mu, logvar


# -----------------------------------------------------------------------------#
# 4) KL utilities
# -----------------------------------------------------------------------------#
def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def kl_freebits_reduce(kl_zdhw: torch.Tensor, free_bits: float = 0.75,
                       group: int = 4) -> torch.Tensor:
    B, Z, d, h, w = kl_zdhw.shape
    kl_Z = kl_zdhw.view(B, Z, -1).sum(-1)  # [B, Z]

    if group > 1:
        if Z % group != 0:
            G = (Z // group) * group
        else:
            G = Z

        if G > 0:
            kl_Z_grouped = kl_Z[:, :G].view(B, G // group, group).sum(-1)
            if G < Z:
                kl_Z_remaining = kl_Z[:, G:].sum(-1, keepdim=True)
                kl_Z = torch.cat((kl_Z_grouped, kl_Z_remaining), dim=1)
            else:
                kl_Z = kl_Z_grouped

    kl_g = torch.clamp(kl_Z, min=free_bits).sum(-1)
    return kl_g.mean()


def beta_warmup(global_step: int, total_steps: int,
                beta_max: float = 0.6, warmup_ratio: float = 0.4) -> float:
    denom = max(1.0, warmup_ratio * total_steps)
    t = min(1.0, global_step / denom)
    return beta_max * t


# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
def dice_loss_with_logits(
        logits: torch.Tensor,                  # [B, C, D, H, W]
        targets: torch.Tensor,                 # [B, C, D, H, W], float in {0,1}
        channel_weights: Optional[Sequence[float]] = None,
        eps: float = 1e-6
) -> torch.Tensor:
    """Implementation details for dice_loss_with_logits."""
    if targets.dtype != torch.float32:
        targets = targets.float()

    probs = torch.sigmoid(logits)  # [B, C, D, H, W]

    B, C, D, H, W = probs.shape
    probs_flat = probs.view(B * C, -1)     # [BC, N]
    targets_flat = targets.view(B * C, -1) # [BC, N]

    intersection = (probs_flat * targets_flat).sum(dim=1)      # [BC]
    sum_probs = probs_flat.sum(dim=1)                          # [BC]
    sum_targets = targets_flat.sum(dim=1)                      # [BC]

    dice = (2.0 * intersection + eps) / (sum_probs + sum_targets + eps)  # [BC]
    loss_per = 1.0 - dice                                                # [BC]

    if channel_weights is not None:
        cw = torch.as_tensor(channel_weights, dtype=logits.dtype, device=logits.device)  # [C]
        if cw.numel() != C:
            raise ValueError(f'Invalid input or configuration.')



        w_bc = cw.repeat(B)  # [BC]

        loss = (loss_per * w_bc).sum() / w_bc.sum().clamp_min(1e-6)
    else:
        loss = loss_per.mean()

    return loss


# -----------------------------------------------------------------------------#
# 6) Dual-Head VAE Loss:

# -----------------------------------------------------------------------------#
def dual_head_vae_loss_function(
        logits_sem: torch.Tensor,    # [B, C_sem, D, H, W]
        logits_edge: Optional[torch.Tensor],   # [B, C_edge(=3), D, H, W] or None
        x_sem_idx: torch.Tensor,     # [B, D, H, W] (long indices)
        x_edge: Optional[torch.Tensor],        # [B, C_edge, D, H, W] (0/1 float) or None
        mu: torch.Tensor,
        logvar: torch.Tensor,        # [B, Z, d, h, w]
        beta: float = 0.0,
        free_bits: float = 0.75,
        group: int = 4,

        class_weights: Optional[torch.Tensor] = None,  # [C_sem] or None

        miss_mask: Optional[torch.Tensor] = None,      # [B,1,D,H,W] or [B,D,H,W]
        miss_lambda: float = 3.0,

        lambda_edge: float = 1.0,
        edge_bce_weight: float = 1.0,
        edge_dice_weight: float = 1.0,
        edge_dice_channel_weights: Optional[Sequence[float]] = None,
):
    """Implementation details for dual_head_vae_loss_function."""
    if x_sem_idx.dtype != torch.long:
        x_sem_idx = x_sem_idx.long()
    if x_edge is not None and x_edge.dtype != torch.float32:
        x_edge = x_edge.float()

    B, C_sem, D, H, W = logits_sem.shape


    if miss_mask is not None:
        if miss_mask.dim() == 5 and miss_mask.size(1) == 1:
            miss_mask_4d = miss_mask[:, 0]  # [B,D,H,W]
        elif miss_mask.dim() == 4:
            miss_mask_4d = miss_mask
        else:
            raise ValueError(f'Invalid input or configuration.')
        weight_miss_sem = (1.0 - miss_mask_4d) + miss_lambda * miss_mask_4d
        weight_miss_edge = weight_miss_sem.unsqueeze(1)  # [B,1,D,H,W]
    else:
        weight_miss_sem = None
        weight_miss_edge = None


    ce_sem_elem = F.cross_entropy(
        logits_sem, x_sem_idx,
        weight=class_weights,
        reduction='none'
    )  # [B,D,H,W]

    if weight_miss_sem is not None:
        ce_sem = (ce_sem_elem * weight_miss_sem).sum() / weight_miss_sem.sum().clamp_min(1.0)
    else:
        ce_sem = ce_sem_elem.mean()


    if logits_edge is not None and x_edge is not None and lambda_edge > 0.0:
        bce_edge_elem = F.binary_cross_entropy_with_logits(
            logits_edge, x_edge,
            reduction='none'
        )  # [B,C_edge,D,H,W]

        if weight_miss_edge is not None:
            bce_edge = (bce_edge_elem * weight_miss_edge).sum() / weight_miss_edge.sum().clamp_min(1.0)
        else:
            bce_edge = bce_edge_elem.mean()

        dice_edge = dice_loss_with_logits(
            logits_edge,
            x_edge,
            channel_weights=edge_dice_channel_weights,
        )

        edge_loss = edge_bce_weight * bce_edge + edge_dice_weight * dice_edge
    else:
        bce_edge = logits_sem.new_zeros(())
        dice_edge = logits_sem.new_zeros(())
        edge_loss = logits_sem.new_zeros(())

    # ------------------ KL + free-bits ------------------ #
    kl_zdhw = kl_per_dim(mu, logvar)
    kl = kl_freebits_reduce(kl_zdhw, free_bits=free_bits, group=group)


    loss = ce_sem + lambda_edge * edge_loss + beta * kl

    return loss, ce_sem, bce_edge, dice_edge, kl


# -----------------------------------------------------------------------------#
# (Optional) Quick self-test
# -----------------------------------------------------------------------------#
if __name__ == "__main__":

    B = 2
    C_sem = 8
    C_edge = 3       # Ch1/Ch2/Ch0
    Z_default = 32
    base_ch_default = 16
    dropout_default = 0.2
    groups_default = 4

    D = H = W = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    x_sem_idx = torch.randint(0, C_sem, (B, D, H, W), device=device)
    x_sem_onehot = F.one_hot(x_sem_idx, num_classes=C_sem).permute(0, 4, 1, 2, 3).float()


    x_edge = (torch.rand(B, C_edge, D, H, W, device=device) < 0.02).float()


    x_in = torch.cat([x_sem_onehot, x_edge], dim=1)  # [B, C_sem + 3, D, H, W]

    print("--- Self-Test: Struct-Aware Dual-Head VAE ---")
    model = AutoencoderKL_3D_DualHead(
        num_semantic_classes=C_sem,
        num_edge_channels=C_edge,
        z_channels=Z_default,
        input_shape=(D, H, W),
        base_ch=base_ch_default,
        groups=groups_default,
        dropout_prob=dropout_default,
    ).to(device)

    x_in = x_in.to(device)
    x_sem_idx = x_sem_idx.to(device)
    x_edge = x_edge.to(device)

    logits_sem, logits_edge, mu, logvar = model(x_in)


    miss_mask = None

    beta = beta_warmup(global_step=100, total_steps=1000,
                       beta_max=0.6, warmup_ratio=0.4)


    edge_dice_channel_weights = [1.0, 2.0, 0.5]

    loss, ce_sem, bce_edge, dice_edge, kl = dual_head_vae_loss_function(
        logits_sem, logits_edge,
        x_sem_idx, x_edge,
        mu, logvar,
        beta=beta, free_bits=0.75, group=groups_default,
        class_weights=None,
        miss_mask=miss_mask, miss_lambda=3.0,
        lambda_edge=1.0,
        edge_bce_weight=1.0,
        edge_dice_weight=1.0,
        edge_dice_channel_weights=edge_dice_channel_weights,
    )

    print(f"[SelfTest] loss={loss.item():.4f}, "
          f"CE_sem={ce_sem.item():.4f}, "
          f"BCE_edge={bce_edge.item():.4f}, "
          f"Dice_edge={dice_edge.item():.4f}, "
          f"KL={kl.item():.4f}, beta={beta:.3f}")
    print(f"[SelfTest] Input shape:  {x_in.shape}")
    print(f"[SelfTest] Sem logits:   {logits_sem.shape}")
    print(f"[SelfTest] Edge logits:  {logits_edge.shape}")
    print(f"[SelfTest] Mu shape:     {mu.shape}")
