from __future__ import annotations

from typing import Optional, Sequence, Tuple

try:
    import jittor as jt
    from jittor import nn
except Exception as exc:  # pragma: no cover
    raise ImportError("jittor is required for repro.jittor_models") from exc

from models_infmae_skip4 import infmae_vit_base_patch16


class PatchEmbedJittor(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def execute(self, x):
        x = self.proj(x)
        x = self.norm(x.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))
        return self.act(x)


class InfMAEBackboneJittor(nn.Module):
    """InfMAE encoder wrapper for Jittor downstream alignment."""

    def __init__(self):
        super().__init__()
        self.encoder = infmae_vit_base_patch16()
        self.out_channels = (256, 384, 768)

    @staticmethod
    def _interpolate_pos_embed(pos_embed, height: int, width: int):
        _, token_len, channels = pos_embed.shape
        source_hw = int(round(token_len ** 0.5))
        if source_hw == height and source_hw == width:
            return pos_embed
        pos = pos_embed.reshape((1, source_hw, source_hw, channels)).transpose((0, 3, 1, 2))
        pos = nn.interpolate(pos, size=(height, width), mode="bilinear", align_corners=False)
        return pos.transpose((0, 2, 3, 1)).reshape((1, height * width, channels))

    def execute(self, x):
        encoder = self.encoder

        x1 = encoder.patch_embed1(x)
        for blk in encoder.blocks1:
            x1 = blk(x1)

        x2 = encoder.patch_embed2(x1)
        for blk in encoder.blocks2:
            x2 = blk(x2)

        x3 = encoder.patch_embed3(x2)

        b, c, h, w = x3.shape
        tokens = x3.reshape((b, c, h * w)).transpose((0, 2, 1))
        tokens = encoder.patch_embed4(tokens)
        tokens = tokens + self._interpolate_pos_embed(encoder.pos_embed, h, w)

        for blk in encoder.blocks3:
            tokens = blk(tokens)
        tokens = encoder.norm(tokens)
        
        x3 = tokens.transpose((0, 2, 1)).reshape((b, c, h, w))
        return x1, x2, x3


class ConvNormActJittor(nn.Module):
    """与 PyTorch ConvNormAct 对齐: Conv → BatchNorm → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm(out_channels),
            nn.Relu(),
        )

    def execute(self, x):
        return self.block(x)


class SimpleUPerHeadJittor(nn.Module):
    """UPerNet head with PPM + FPN."""

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 384, 768),
        channels: int = 128,
        num_classes: int = 9,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.pool_scales = tuple(pool_scales)

        self.lateral_convs = nn.ModuleList([ConvNormActJittor(c, channels, kernel_size=1) for c in in_channels])
        self.fpn_convs = nn.ModuleList([ConvNormActJittor(channels, channels, kernel_size=3) for _ in in_channels])

        self.ppm_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((s, s)) for s in self.pool_scales])
        self.ppm_convs = nn.ModuleList([ConvNormActJittor(in_channels[-1], channels, kernel_size=1) for _ in self.pool_scales])
        self.ppm_fuse = ConvNormActJittor(in_channels[-1] + len(self.pool_scales) * channels, channels, kernel_size=3)

        self.fuse = ConvNormActJittor(channels * len(in_channels), channels, kernel_size=3)
        self.classifier = nn.Conv(channels, num_classes, kernel_size=1, stride=1)

    def execute(self, feats: Sequence, output_size: Optional[Tuple[int, int]] = None):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        psp_in = feats[-1]
        psp_outs = [psp_in]
        for pool, conv in zip(self.ppm_pools, self.ppm_convs):
            pooled = pool(psp_in)
            pooled = conv(pooled)
            pooled = nn.interpolate(pooled, size=psp_in.shape[2:], mode="bilinear", align_corners=False)
            psp_outs.append(pooled)
        laterals[-1] = self.ppm_fuse(jt.concat(psp_outs, dim=1))

        for i in range(len(laterals) - 2, -1, -1):
            up = nn.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="bilinear", align_corners=False)
            laterals[i] = laterals[i] + up

        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        target_size = fpn_outs[0].shape[2:]
        resized = [nn.interpolate(x, size=target_size, mode="bilinear", align_corners=False) for x in fpn_outs]

        fused = self.fuse(jt.concat(resized, dim=1))
        logits = self.classifier(fused)
        if output_size is not None:
            logits = nn.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


class InfMAEDownstreamJittor(nn.Module):
    def __init__(self, num_classes: int = 9, freeze_backbone: bool = True, channels: int = 128):
        super().__init__()
        self.backbone = InfMAEBackboneJittor()
        self.decode_head = SimpleUPerHeadJittor(num_classes=num_classes, channels=channels)
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.stop_grad()

    def execute(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats, output_size=(x.shape[2], x.shape[3]))


class InfMAEMSEPretrainJittor(nn.Module):
    """Pretraining wrapper around the full InfMAE MAE model."""

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.model = infmae_vit_base_patch16()
        self.mask_ratio = mask_ratio

    def execute(self, x):
        """返回 (loss, pred, mask)。"""
        loss, pred, mask = self.model(x, mask_ratio=self.mask_ratio)
        return loss, pred, mask
