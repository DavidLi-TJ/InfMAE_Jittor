
import numpy as np
import jittor as jt


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 从 2D 网格坐标计算正弦余弦位置编码
    assert embed_dim % 2 == 0, \
        f"embed_dim 非偶 {embed_dim}"
    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    #从 1D 坐标向量计算正弦余弦位置编码
    assert embed_dim % 2 == 0, \
        f"embed_dim 非偶 {embed_dim}"
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  #  (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    # 检查检查点中是否包含位置编码
    if 'pos_embed' in checkpoint_model:
        # 获取检查点中的位置编码
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        
        # 获取位置编码的特征维度（最后一维）
        embedding_size = pos_embed_checkpoint.shape[-1]
        
        # 获取目标模型的 patch 数量
        num_patches = model.patch_embed.num_patches
        
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        
        # 计算新分辨率（目标模型的网格大小）
        new_size = int(num_patches ** 0.5)
        
        # 只在尺寸不同时执行插值
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (
                orig_size, orig_size, new_size, new_size))
            
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            
            pos_tokens = jt.nn.interpolate(
                pos_tokens, 
                size=(new_size, new_size), 
                mode='bicubic', 
                align_corners=False
            )
            
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            
            new_pos_embed = jt.concat([extra_tokens, pos_tokens], dim=1)
            
            checkpoint_model['pos_embed'] = new_pos_embed
