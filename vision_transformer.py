# -*- coding: utf-8 -*-

import jittor as jt
import jittor.nn as nn
from functools import partial
import math


# ============================================================
# 常量定义 - ImageNet 数据集标准化参数
# ============================================================

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)

IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# ============================================================
# 辅助函数
# ============================================================

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    jt.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


# ============================================================
# DropPath 类 - 随机深度（Stochastic Depth）
# ============================================================

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        
        keep_prob = 1 - self.drop_prob
        # 生成与输入形状兼容的随机张量
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
        random_tensor = jt.floor(random_tensor)  # 二值化: 0 或 1
        # 缩放以保持期望值
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================
# 配置辅助函数
# ============================================================

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs
    }


# ============================================================
# 预训练模型配置字典
# ============================================================

default_cfgs = {
    # -------------------------------------------------------
    # 纯 Patch 模型配置
    # -------------------------------------------------------
    
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),

    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),

    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),

    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),

    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),

    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),

    'vit_huge_patch16_224': _cfg(),

    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),

    # -------------------------------------------------------
    # 混合模型配置（CNN Backbone + Transformer）
    # -------------------------------------------------------

    'vit_small_resnet26d_224': _cfg(),

    'vit_small_resnet50d_s3_224': _cfg(),

    'vit_base_resnet26d_224': _cfg(),

    'vit_base_resnet50d_224': _cfg(),
}


# ============================================================
# CMlp 类 - 卷积 MLP 模块
# ============================================================

class CMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        # 设置输出和隐藏层维度
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 第一个 1x1 卷积层：升维
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        
        # 激活函数
        self.act = act_layer()
        
        # 第二个 1x1 卷积层：降维
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        
        # Dropout 层
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        # 第一层：线性变换 + 激活 + Dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        # 第二层：线性变换 + Dropout
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


# ============================================================
# Mlp 类 - 标准 MLP 模块
# ============================================================

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        # 设置维度参数
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 第一个全连接层：升维（通常扩展 4 倍）
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 激活函数（通常使用 GELU）
        self.act = act_layer()
        
        # 第二个全连接层：降维
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Dropout 层
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        # 第一层：升维 + 激活 + Dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        # 第二层：降维 + Dropout
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


# ============================================================
# CBlock 类 - 卷积 Transformer Block
# ============================================================

class CBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 第一个 Layer Norm（Pre-Norm 架构）
        self.norm1 = norm_layer(dim)
        
        # 1×1 卷积：通道混合
        self.conv1 = nn.Conv2d(dim, dim, 1)
        
        # 1×1 卷积：输出投影
        self.conv2 = nn.Conv2d(dim, dim, 1)
        
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 第二个 Layer Norm
        self.norm2 = norm_layer(dim)
        
        # MLP 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # 卷积 MLP 模块
        self.mlp = CMlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )

    def execute(self, x, mask=None):
        x_normed = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 1×1 卷积：通道变换
        x_conv = self.conv1(x_normed)
        
        # 如果提供了掩码，应用掩码（用于 MAE 掩码重建任务）
        if mask is not None:
            x_conv = mask * x_conv
        
        # 5×5 深度可分离卷积：局部注意力
        x_attn = self.attn(x_conv)
        
        # 1×1 卷积：输出投影
        x_proj = self.conv2(x_attn)
        
        # 残差连接 + 随机深度
        x = x + self.drop_path(x_proj)
        
        # ===== MLP 子层 =====
        # Layer Norm
        x_normed = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 卷积 MLP + 残差连接 + 随机深度
        x = x + self.drop_path(self.mlp(x_normed))
        
        return x


# ============================================================
# Attention 类 - 多头自注意力机制
# ============================================================

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        
        # 注意力头数量
        self.num_heads = num_heads
        
        # 每个头的维度
        head_dim = dim // num_heads
        self.head_dim = head_dim
        
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 注意力权重的 Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        
        # 输出的 Dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x)
        
        # 重塑并分割为 Q, K, V
        # (B, N, 3*C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # 分离 Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 各自形状: (B, num_heads, N, head_dim)
        
        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        
        # ===== 步骤 4: 注意力 Dropout =====
        attn = self.attn_drop(attn)
        
        # (B, heads, N, N) @ (B, heads, N, head_dim) -> (B, heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # ===== 步骤 6 & 7: 输出投影 + Dropout =====
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ============================================================
# Block 类 - 标准 Transformer Encoder Block
# ============================================================

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 第一个 Layer Norm（用于注意力子层）
        self.norm1 = norm_layer(dim)
        
        # 多头自注意力模块
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 第二个 Layer Norm（用于 MLP 子层）
        self.norm2 = norm_layer(dim)
        
        # MLP 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # MLP 模块
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )

    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


# ============================================================
# PatchEmbed_F 类 - 图像到 Patch 嵌入
# ============================================================

class PatchEmbed_F(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        # 确保尺寸是元组形式
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        # 计算 patch 数量
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        # 保存配置信息
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        B, C, H, W = x.shape

        if H != self.img_size[0] or W != self.img_size[1]:
            x = nn.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        # 卷积: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        x = x.flatten(2)
        
        # 转置: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.transpose(1, 2)
        
        return x


# ============================================================
# PatchEmbed 类 - 图像到 Patch 嵌入（带归一化和激活）
# ============================================================

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        # 转换为元组
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        # 计算 patch 数量
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        # 保存配置
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Patch 嵌入卷积
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Layer Normalization
        # 对每个位置的嵌入向量进行归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        self.act = nn.GELU()

    def execute(self, x):
        B, C, H, W = x.shape

        if H != self.img_size[0] or W != self.img_size[1]:
            x = nn.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        # Patch 嵌入卷积
        x = self.proj(x)
        
        # Layer Normalization
        # Conv2d 输出是 (B, C, H, W)，LayerNorm 需要 (B, H, W, C)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # GELU 激活
        return self.act(x)


# ============================================================
# HybridEmbed 类 - 混合 CNN-Transformer 嵌入
# ============================================================

class HybridEmbed(nn.Module):

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        
        # 验证 backbone 类型
        assert isinstance(backbone, nn.Module), "backbone must be an nn.Module"
        
        # 转换图像尺寸为元组
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        
        if feature_size is None:
            with jt.no_grad():
                # 保存原始训练状态
                training = backbone.training
                if training:
                    backbone.eval()
                
                o = self.backbone(jt.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                
                # 获取特征图的 spatial 尺寸
                feature_size = o.shape[-2:]
                
                # 获取通道数
                feature_dim = o.shape[1]
                
                # 恢复原始训练状态
                backbone.train(training)
        else:
            # 使用提供的特征图尺寸
            feature_size = to_2tuple(feature_size)
            
            # 从 backbone 的元数据获取通道数
            feature_dim = self.backbone.feature_info.channels()[-1]
        
        # 计算 patch 数量（即特征图的空间位置数）
        self.num_patches = feature_size[0] * feature_size[1]
        
        # 线性投影层：将 CNN 特征维度映射到 Transformer 嵌入维度
        self.proj = nn.Linear(feature_dim, embed_dim)

    def execute(self, x):
        # CNN 特征提取
        x = self.backbone(x)[-1]
        
        # 展平特征图: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # 线性投影
        x = self.proj(x)
        
        return x


# ============================================================
# ConvViT 类 - 卷积 Vision Transformer 主模型
# ============================================================

class ConvViT(nn.Module):

    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0., 
        hybrid_backbone=None, 
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        # 分类相关属性
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # ===== 构建 Patch Embedding =====
        if hybrid_backbone is not None:
            # 使用混合 CNN-Transformer 嵌入
            self.patch_embed = HybridEmbed(
                hybrid_backbone, 
                img_size=img_size, 
                in_chans=in_chans, 
                embed_dim=embed_dim
            )
        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size[0], 
                patch_size=patch_size[0], 
                in_chans=in_chans, 
                embed_dim=embed_dim[0]
            )
            
            self.patch_embed2 = PatchEmbed(
                img_size=img_size[1], 
                patch_size=patch_size[1], 
                in_chans=embed_dim[0], 
                embed_dim=embed_dim[1]
            )
            
            self.patch_embed3 = PatchEmbed(
                img_size=img_size[2], 
                patch_size=patch_size[2], 
                in_chans=embed_dim[1], 
                embed_dim=embed_dim[2]
            )
            
            # 第三级之后的线性投影（可选）
            num_patches = self.patch_embed3.num_patches
            self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        
        self.pos_embed = nn.Parameter(jt.zeros(1, num_patches, embed_dim[2]))
        
        # Dropout 层
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depth))]
        
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio[0], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer
            )
            for i in range(depth[0])
        ])
        
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio[1], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[depth[0] + i], 
                norm_layer=norm_layer
            )
            for i in range(depth[1])
        ])
        
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio[2], 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[depth[0] + depth[1] + i], 
                norm_layer=norm_layer
            )
            for i in range(depth[2])
        ])
        
        self.norm = norm_layer(embed_dim[-1])
        
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        
        # 对所有子模块应用自定义初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 线性层：截断正态分布初始化
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                # 偏置初始化为零
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm: 标准初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        
        return self.head#获取分类器层，返回:nn.Module: 模型的分类头（通常是最后的线性层）

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        
        # ===== 第一级：细粒度特征提取 =====
        # Patch Embedding: (B, 3, H, W) -> (B, embed_dim[0], H1, W1)
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        
        # 通过第一级卷积 Blocks
        for blk in self.blocks1:
            x = blk(x)
        
        x = self.patch_embed2(x)
        
        # 通过第二级卷积 Blocks
        for blk in self.blocks2:
            x = blk(x)
        
        x = self.patch_embed3(x)
        
        # 展平为序列: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过标准 Transformer Blocks
        for blk in self.blocks3:
            x = blk(x)
        
        # 最终归一化
        x = self.norm(x)
        
        return x.mean(1)

    def execute(self, x):
        # 特征提取
        x = self.forward_features(x)
        
        # 分类
        x = self.head(x)
        
        return x
