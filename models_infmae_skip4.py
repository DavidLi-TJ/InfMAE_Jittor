from functools import partial
import numpy as np
import jittor as jt
import jittor.nn as nn

from vision_transformer import PatchEmbed, Block, CBlock, PatchEmbed_F

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderInfMAE(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # =====================================================================
        # 第一部分：编码器 (Encoder) 定义
        # =====================================================================

        self.patch_embed = PatchEmbed_F(
            img_size[0],
            patch_size[0] * patch_size[1] * patch_size[2],
            in_chans,
            embed_dim[2]
        )

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

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])

        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)

        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches

        # pos_embed 形状: (1, num_patches, embed_dim[2])
        #                 = (1, 196, 768)
        # requires_grad=False: 冻结参数，使用预计算的 sin-cos 编码
        self.pos_embed = nn.Parameter(
            jt.zeros([1, num_patches, embed_dim[2]]),
            requires_grad=False
        )

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[0],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer
            )
            for i in range(depth[0])
        ])

        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[1],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer
            )
            for i in range(depth[1])
        ])

        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[2],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer
            )
            for i in range(depth[2])
        ])

        self.norm = norm_layer(embed_dim[-1])

        # =====================================================================
        # 第二部分：解码器 (Decoder) 定义
        # =====================================================================

        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(jt.zeros([1, 1, decoder_embed_dim]))

        self.decoder_pos_embed = nn.Parameter(
            jt.zeros([1, num_patches, decoder_embed_dim]),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio[0],
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer
            )
            for i in range(decoder_depth)
        ])

        # --- 解码器归一化层 ---
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (patch_size[0] * patch_size[1] * patch_size[2]) ** 2 * in_chans,
            bias=True
        )

        # =====================================================================
        # 第三部分：其他配置
        # =====================================================================

        # 是否使用归一化像素损失
        self.norm_pix_loss = norm_pix_loss

        # 初始化模型权重
        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed3.num_patches ** .5),
            cls_token=False
        )

        self.pos_embed.assign(jt.array(pos_embed.astype(np.float32)).unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed3.num_patches ** .5),
            cls_token=False
        )
        self.decoder_pos_embed.assign(jt.array(decoder_pos_embed.astype(np.float32)).unsqueeze(0))

        w = self.patch_embed3.proj.weight

        jt.nn.init.xavier_uniform_(w.reshape([w.shape[0], -1]))

        jt.init.gauss_(self.mask_token, mean=0.0, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            jt.nn.init.xavier_uniform_(m.weight)

            # 如果有偏置项，初始化为零
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            # LayerNorm: 标准初始化
            # weight (scale) = 1.0, bias (shift) = 0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = 16

        h_img, w_img = imgs.shape[2], imgs.shape[3]
        side = min(h_img, w_img)
        side = side - (side % p)

        if side == 0:
            imgs = nn.interpolate(imgs, size=(p, p), mode='bilinear', align_corners=False)
            side = p
        elif h_img != side or w_img != side:
            imgs = imgs[:, :, :side, :side]

        h = w = side // p

        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)

        x = jt.einsum('nchpwq->nhwpqc', x)

        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def unpatchify(self, x):
        # Patch 大小
        p = 16

        h = w = int(x.shape[1] ** .5)
        token_count = h * w
        if token_count != x.shape[1]:
            x = x[:, :token_count, :]

        x = x.reshape(x.shape[0], h, w, p, p, 3)

        # 第二步：使用 einsum 重排维度（patchify 的逆操作）
        # 'nhwpqc' → 'nchpwq'
        x = jt.einsum('nhwpqc->nchpwq', x)

        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    def random_masking(self, x, mask_ratio):
        # 获取批次大小
        N = x.shape[0]

        L = self.patch_embed3.num_patches

        x_ = jt.mean(x, dim=-1)

        ids_shuffle = jt.argsort(x_, 1)[0]

        ids_restore = jt.argsort(ids_shuffle, 1)[0]

        ids_keep = ids_shuffle[:, ::4]

        mask = jt.ones([N, L])

        mask[:, ::4] = 0

        mask = jt.gather(mask, 1, ids_restore)  # 与 PyTorch 一致: (input, dim, index)

        # 返回三个值供后续使用
        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        x_ = self.patch_embed(x)

        ids_keep, mask, ids_restore = self.random_masking(x_, mask_ratio)


        mask_for_patch1 = (
            mask.reshape(-1, 14, 14)
            .unsqueeze(-1)
            .repeat(1, 1, 1, 16)
            .reshape(-1, 14, 14, 4, 4)
            .permute(0, 1, 3, 2, 4)
            .reshape(x.shape[0], 56, 56)
            .unsqueeze(1)
        )

        mask_for_patch2 = (
            mask.reshape(-1, 14, 14)
            .unsqueeze(-1)
            .repeat(1, 1, 1, 4)
            .reshape(-1, 14, 14, 2, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(x.shape[0], 28, 28)
            .unsqueeze(1)
        )


        # Step 3.1: Patch Embedding
        x = self.patch_embed1(x)

        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)

        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)


        x = self.patch_embed2(x)

        # Step 4.2: 卷积块处理（带掩码）
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)

        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)


        # Step 5.1: Final Patch Embedding
        x = self.patch_embed3(x)

        # Step 5.2: 转换为序列格式
        # flatten(2): (N, 768, 14, 14) → (N, 768, 196)
        # permute(0, 2, 1): (N, 768, 196) → (N, 196, 768)
        x = x.flatten(2).permute(0, 2, 1)

        x = self.patch_embed4(x)

        x = x + self.pos_embed

        x = jt.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        stage1_embed = jt.gather(
            stage1_embed, 1,
            ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]),
        )
        stage2_embed = jt.gather(
            stage2_embed, 1,
            ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]),
        )

        # ===== 阶段 6: Transformer 编码器（主处理）=====

        for blk in self.blocks3:
            x = blk(x)

        x = x + stage1_embed + stage2_embed

        x = self.norm(x)

        # 返回编码结果、掩码和恢复索引
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] - x.shape[1],
            1
        )

        # 保存当前 x 的引用（可见 token 的嵌入）
        x_ = x

        insert_interval = 3  # 每隔几个位置插入一组 mask tokens

        insert_count = mask_tokens.shape[1] // insert_interval

        result_shape = (x_.shape[0], x_.shape[1] + insert_count * insert_interval, x_.shape[2])

        result = jt.empty(*result_shape)

        # 执行循环插入
        for i in range(insert_count):
            start_idx = i * (insert_interval + 1)
            end_idx = start_idx + insert_interval

            if start_idx == 0:
                result[:, :1, :] = x_[:, i, :].unsqueeze(1)

                # result[1:4] = 第 0 组的 3 个 mask tokens
                result[:, start_idx + 1:end_idx + 1, :] = \
                    mask_tokens[:, i * insert_interval:(i + 1) * insert_interval, :]
            else:
                result[:, start_idx:start_idx + 1, :] = x_[:, i, :].unsqueeze(1)

                # result[start_idx+1:end_idx+1] = 第 i 组的 3 个 mask tokens
                result[:, start_idx + 1:end_idx + 1, :] = \
                    mask_tokens[:, i * insert_interval:(i + 1) * insert_interval, :]

        x = jt.gather(result, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        # ===== 步骤 7: 解码器输出归一化 =====
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):

        target = self.patchify(imgs)

        # ===== 步骤 2: 可选的像素归一化 =====
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)

            # 计算每个 patch 的方差
            # var: (N, L, 1)
            var = target.var(dim=-1, keepdim=True)

            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2

        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()

        return loss

    def execute(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        pred = self.forward_decoder(latent, ids_restore)

        loss = self.forward_loss(imgs, pred, mask)

        # 返回三元组：损失、预测、掩码
        return loss, pred, mask

def infmae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderInfMAE(
        # ===== 编码器配置 =====
        img_size=[224, 56, 28],           # 三级分辨率
        patch_size=[4, 2, 2],             # 三级 patch 大小
        embed_dim=[256, 384, 768],        # 三级嵌入维度
        depth=[2, 2, 11],                 # 各 stage 深度
        num_heads=12,                     # 注意力头数

        # ===== 解码器配置 =====
        decoder_embed_dim=512,            # 解码器维度
        decoder_depth=2,                  # 解码器深度（8→2 以加速实验）
        decoder_num_heads=16,             # 解码器注意力头数

        # ===== 其他配置 =====
        mlp_ratio=[4, 4, 4],              # MLP 扩展倍数
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 归一化层
        **kwargs                           # 额外参数
    )
    return model

infmae_vit_base_patch16 = infmae_vit_base_patch16_dec512d8b

# 解码器配置: 512 维嵌入, 8 个 block（默认配置的完整版本）
