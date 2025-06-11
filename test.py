import math
import time
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from multiprocessing import Pool
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from flash_attn.modules import mha
from adabelief_pytorch import AdaBelief

# \b#.+\b\n 
OUTPUT_CHANNELS = 3
device = "cuda"

class PatchEmbed(nn.Module):
    """
    将输入图像分割成不重叠的 Patch, 并对每个 Patch 进行线性投影到嵌入空间。
    
    参数:
        img_size (int or tuple[int]): 输入图像的高度和宽度（默认为 224)。
        patch_size (int or tuple[int]): 每个 Patch 的尺寸（默认为 4)。
        in_chans (int): 输入通道数（默认为 3)。
        embed_dim (int): 输出嵌入维数（默认为 96)。
        norm_layer (callable, optional): 用于对嵌入向量进行归一化的层。
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # 确保 img_size 与 patch_size 均为元组形式
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        # 计算 Patch 的分辨率和数量
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 使用卷积实现 Patch 划分和投影（卷积核大小和步幅均为 patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """
        前向传播:
            1. 使用卷积层对输入进行 Patch 分割及投影；
            2. 将卷积输出展平成序列, 并进行维度置换；
            3. 如存在归一化层, 则对得到的嵌入序列进行归一化。
        
        参数:
            x (Tensor): 输入张量, 形状为 [B, C, H, W]
        
        返回:
            Tensor: 输出嵌入序列, 形状为 [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        
        # 卷积投影后展平并交换通道顺序
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """
        计算 PatchEmbed 层的 FLOPs 数量。
        
        返回:
            int: FLOPs 数量（未考虑 batch_size)
        """
        Ho, Wo = self.patches_resolution
        conv_flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        norm_flops = Ho * Wo * self.embed_dim if self.norm is not None else 0
        return conv_flops + norm_flops
    

class Mlp(nn.Module):
    """
    多层感知机模块（MLP），包含两层全连接、激活函数和 dropout。
    
    参数:
        in_features (int): 输入特征维度。
        hidden_features (Optional[int]): 隐藏层特征维度（默认与 in_features 相同）。
        out_features (Optional[int]): 输出特征维度（默认与 in_features 相同）。
        act_layer (Callable[..., nn.Module]): 激活函数层，默认为 nn.GELU。
        drop (float): dropout 概率，默认为 0.0。
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, act_layer: Callable[..., nn.Module] = nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    局部窗口内的多头注意力机制，
    包含相对位置编码以及缩放点积注意力计算。
    """
    def __init__(self, dim: int, window_size: tuple, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0., pretrained_window_size: list = [0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size   # (height, width)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # learnable scale for attention logits
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        
        # MLP用于生成相对位置偏置
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # 构造相对位置坐标表, shape: (1, 2*H-1, 2*W-1, 2)
        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]), dim=-1)
        relative_coords_table = relative_coords_table.unsqueeze(0).contiguous()
        
        # 归一化处理
        if pretrained_window_size[0] > 0:
            relative_coords_table[..., 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[..., 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[..., 0] /= (window_size[0] - 1)
            relative_coords_table[..., 1] /= (window_size[1] - 1)
        relative_coords_table = relative_coords_table * 8
        # 使用对数缩放，log2(8)==3
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        self.register_buffer("relative_coords_table", relative_coords_table)

        # 计算窗口内每个 token 的相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=0)  # shape: (2, H, W)
        coords_flatten = torch.flatten(coords, 1)  # shape: (2, H*W)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, H*W, H*W)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (H*W, H*W, 2)
        relative_coords[..., 0] += window_size[0] - 1
        relative_coords[..., 1] += window_size[1] - 1
        relative_coords[..., 0] *= (2 * window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 线性变换用于计算q, k, v
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播:
          1. 计算 q, k, v 及其重排；
          2. 归一化后计算注意力，并通过logit_scale进行缩放；
          3. 添加相对位置偏置，若有mask则结合mask；
          4. 输出经过投影处理后的张量.
        """
        B, N, C = x.shape
        
        # 构造 qkv 的偏置（若启用 qkv_bias）
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数计算，先对 q 和 k 进行归一化
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        max_scale = torch.log(torch.tensor(1. / 0.01, device=attn.device))
        logit_scale = torch.clamp(self.logit_scale, max=max_scale).exp()
        attn = attn * logit_scale

        # 计算相对位置偏置，并通过sigmoid归一化后放大16倍
        bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_bias = bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.num_heads)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        relative_bias = 16 * torch.sigmoid(relative_bias)
        attn = attn + relative_bias.unsqueeze(0)

        # 应用掩码（如果提供）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 将注意力作用到 v 并输出投影结果
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N: int) -> int:
        flops = 0
        # qkv 线性层计算 FLOPs
        flops += N * self.dim * 3 * self.dim
        # 注意力矩阵计算 FLOPs
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # 投影层 FLOPs
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """
    SwinTransformerBlock 实现局部窗口内注意力和多层感知机两部分，
    并根据 shift_size 对输入进行周期性滚动，支持滑动窗口注意力计算。
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (height, width)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 若窗口大于输入尺寸，则不进行平移且窗口大小取最小分辨率
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size 必须在 [0, window_size) 范围内"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 计算注意力掩码，仅在 shift_size > 0 时需要
        if self.shift_size > 0:
            H, W = self.input_resolution
            attn_mask = self._create_mask(H, W)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def _create_mask(self, H: int, W: int) -> torch.Tensor:
        """创建注意力掩码，通过对图像区域划分不同编号后计算差值"""
        img_mask = torch.zeros((1, H, W, 1))
        # 定义高度和宽度的切片，分为3个部分
        h_slices = [slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)]
        w_slices = [slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)]
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # 分块并计算各块之间的差值
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播：
         1. 将输入展平为 (B, H, W, C)；
         2. 根据 shift_size 对输入进行滚动 (shifted_x)；
         3. 利用 window_partition 划分为若干窗口，并计算局部注意力；
         4. 将窗口内的注意力结果反向还原 (window_reverse)；
         5. 恢复原始排列后依次加上残差、归一化和 MLP 模块。
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征数量与分辨率不匹配"
        shortcut = x
        x = x.view(B, H, W, C)

        # 对输入进行平移（若 shift_size > 0）
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 划分窗口后计算注意力
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 还原滚动操作
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # 加残差连接和 MLP 模块
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")

    def flops(self) -> int:
        """计算 FLOPs 数量（不考虑 batch_size）"""
        H, W = self.input_resolution
        flops = self.dim * H * W  # norm1
        nW = (H * W) / (self.window_size * self.window_size)
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio  # MLP部分
        flops += self.dim * H * W  # norm2
        return flops


class AttnBlock6_b(nn.Module):
    """
    通用的注意力块实现，用于对输入特征进行注意力计算。
    
    支持两种模式：
      1. 池化模式（need_pool=True）：采用局部与全局注意力混合计算；
      2. 非池化模式（need_pool=False）：可选择使用 flash-attention 或分头卷积计算注意力。
    
    参数:
      in_channels (int): 输入通道数。
      size (int): 输入特征的尺寸。
      min_head (int): 当非池化模式时使用的最小注意力头数。
      fp16 (bool): 是否启用混合精度计算。
      patched_size (int): Patch 的尺寸。
      force_bmm_batch (bool): 是否强制使用 bmm 批处理。
      test (bool): 是否处于测试模式。
      use_MultiHeadSelfAttention (bool): 非池化模式下是否使用 flash-attention。
      svd (bool): 是否使用 SVD 版本（预留参数）。
    """
    def __init__(self, in_channels, size, min_head=4, fp16=False, patched_size=14, force_bmm_batch=False, test=False,
                 use_MultiHeadSelfAttention=True, svd=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.size = size
        self.force_bmm_batch = force_bmm_batch
        self.test = test
        self.use_MultiHeadSelfAttention = use_MultiHeadSelfAttention
        self.svd = svd

        # 计算 patch 分割相关参数
        self.n = int(size // patched_size * 2)
        self.big_n = int(size // patched_size * 4)
        # 当 n > 1 时采用池化模式，并禁用多头自注意力
        if self.n > 1:
            self.need_pool = True
            self.head = self.n ** 2
            self.use_MultiHeadSelfAttention = False
        else:
            self.need_pool = False
            self.head = min_head

        # 通用归一化层
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.norm_2 = nn.BatchNorm2d(num_features=in_channels)

        if self.need_pool:
            # 池化模式下，使用单层卷积生成 q, k, v
            self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.patch_size = patched_size
            self.split_head_num = (self.size ** 2) // (self.patch_size ** 2)
            self.target_global = patched_size * 4
            self.avg_p = nn.AdaptiveAvgPool2d(output_size=(self.target_global, self.target_global))
            self.up_sample = nn.Upsample(size=(self.size, self.size), mode='bilinear', align_corners=False)
            self.patch_weight = 0.75
            self.global_weight = 1 - self.patch_weight
            self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        else:
            # 非池化模式下
            if self.use_MultiHeadSelfAttention:
                self.small_attention = mha.MHA(embed_dim=size ** 2, num_heads=min_head, device='cuda',
                                               use_flash_attn=False, dwconv=True)
            else:
                # 为每个 head 创建独立卷积，并使用 ModuleList 管理
                conv_dtype = torch.bfloat16 if self.mixed_precision else torch.float
                self.q_group = nn.ModuleList([
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, dtype=conv_dtype) for _ in range(self.head)
                ])
                self.k_group = nn.ModuleList([
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, dtype=conv_dtype) for _ in range(self.head)
                ])
                self.v_group = nn.ModuleList([
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, dtype=conv_dtype) for _ in range(self.head)
                ])
                self.small_proj_out = nn.Conv3d(self.head, 1, kernel_size=1, padding=0, bias=False)

    def cal_bmm_patch(self, q, k, v, b, c):
        """局部注意力计算（池化模式下），处理每个 patch 的注意力"""
        if self.mixed_precision:
            q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        patch_area = self.patch_size ** 2
        q = q.view(b, c * self.split_head_num, patch_area).permute(0, 2, 1)
        k = k.view(b, c * self.split_head_num, patch_area)
        w_in = torch.bmm(q, k) * ((c * self.split_head_num) ** -0.5)
        w_in = F.softmax(w_in, dim=2, dtype=torch.float32)
        v = v.view(b, c * self.split_head_num, patch_area)
        w_in = w_in.permute(0, 2, 1)
        h_in = torch.bmm(v, w_in)
        return h_in.view(b, c, self.size, self.size)

    def cal_bmm_global(self, q, k, v, b, c):
        """全局注意力计算（池化模式下），处理全局注意力信息"""
        if self.mixed_precision:
            q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
        global_area = self.target_global ** 2
        q = q.view(b, c, global_area).permute(0, 2, 1)
        k = k.view(b, c, global_area)
        w_in = torch.bmm(q, k) * (c ** -0.5)
        w_in = F.softmax(w_in, dim=2, dtype=torch.float32)
        v = v.view(b, c, global_area)
        w_in = w_in.permute(0, 2, 1)
        h_in = torch.bmm(v, w_in).view(b, c, self.target_global, self.target_global)
        return self.up_sample(h_in)

    def cal_bmm(self, q, k, v, b, c, h, w, d, h_dim, w_dim):
        """
        逐头注意力计算（非池化模式下），
        将输入 reshape 后在批量上执行矩阵乘法计算注意力。
        """
        q = q.permute(0, 2, 1, 3, 4).reshape(b * d, c, h_dim * w_dim).permute(0, 2, 1)
        k = k.permute(0, 2, 1, 3, 4).reshape(b * d, c, h_dim * w_dim)
        w_in = F.softmax(torch.bmm(q, k) * (c ** -0.5), dim=2, dtype=torch.float32)
        v = v.permute(0, 2, 1, 3, 4).reshape(b * d, c, h_dim * w_dim)
        w_in = w_in.permute(0, 2, 1)
        h_in = torch.bmm(v, w_in).reshape(b, d, c, h_dim, w_dim)
        return h_in

    def call_function(self, x):
        b, c, h, w = x.shape
        if self.need_pool:
            # 池化模式：生成 q, k, v 后计算局部与全局注意力
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            q_p = self.avg_p(q)
            k_p = self.avg_p(k)
            v_p = self.avg_p(v)
            if self.mixed_precision:
                with autocast(dtype=torch.float):
                    h_in_patch = self.cal_bmm_patch(q, k, v, b, c)
                    h_in_global = self.cal_bmm_global(q_p, k_p, v_p, b, c)
            else:
                h_in_patch = self.cal_bmm_patch(q, k, v, b, c)
                h_in_global = self.cal_bmm_global(q_p, k_p, v_p, b, c)
            h_in = self.patch_weight * h_in_patch + self.global_weight * h_in_global
            if self.mixed_precision:
                with autocast(dtype=torch.float):
                    return self.proj_out(h_in)
            else:
                return self.proj_out(h_in)
        else:
            # 非池化模式
            if self.use_MultiHeadSelfAttention:
                # 使用 flash-attention 计算注意力
                h_in = self.small_attention(x)
                return h_in.reshape(b, c, h, w)
            else:
                # 分头卷积计算注意力
                q = x
                k = x
                v = x
                q_list = [layer(q) for layer in self.q_group]
                k_list = [layer(k) for layer in self.k_group]
                v_list = [layer(v) for layer in self.v_group]
                q_stack = torch.stack(q_list, dim=2)
                k_stack = torch.stack(k_list, dim=2)
                v_stack = torch.stack(v_list, dim=2)
                h_in = self.cal_bmm(q_stack, k_stack, v_stack, b, c, h, w, self.head, h, w)
                if self.mixed_precision:
                    with autocast(dtype=torch.float):
                        proj = self.small_proj_out(h_in)
                else:
                    proj = self.small_proj_out(h_in)
                return proj.reshape(b, c, h, w)

    def forward(self, x):
        # 对输入进行归一化后计算注意力，再进行残差连接
        x_norm = self.norm(x)
        if self.mixed_precision:
            with autocast(dtype=torch.bfloat16):
                h_out = self.call_function(x_norm)
        else:
            h_out = self.call_function(x_norm)
        return x + h_out


def unflatten(flattened, num_channels):
    bs, n, p = flattened.size()
    unflattened = torch.reshape(flattened, (bs, n, num_channels, int(np.sqrt(p // num_channels)), int(np.sqrt(p // num_channels))))
    return unflattened


def unpatch(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch == num_channels
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack([torch.cat([patch for patch in x.reshape(batch_size, elem_per_axis, elem_per_axis, ch, h, w)[i]], dim=-2)
                                  for i in range(batch_size)], dim=0)
    restored_images = torch.stack([torch.cat([patch for patch in patches_middle[i]], dim=-1)
                                   for i in range(batch_size)], dim=0).reshape(batch_size, 1, ch, h * elem_per_axis, w * elem_per_axis)
    return restored_images


def unpatch2(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch == num_channels
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack([torch.cat([patch for patch in x.reshape(batch_size, elem_per_axis, elem_per_axis, ch, h, w)[i]], dim=-2)
                                  for i in range(batch_size)], dim=0)
    restored_images = torch.stack([torch.cat([patch for patch in patches_middle[i]], dim=-1)
                                   for i in range(batch_size)], dim=0).reshape(batch_size, ch, h * elem_per_axis, w * elem_per_axis)
    return restored_images, patches_middle


class Attn_Conv2d(nn.Module):
    """
    带注意力机制的卷积层模块。

    参数:
      in_channels (int): 输入通道数。
      out_channels (int): 输出通道数。
      size (int or tuple): 卷积核大小。
      apply_batchnorm (bool): 是否应用归一化层，默认为 True。
      need_attention (bool): 是否添加注意力模块，默认为 False。
      attn_size (int): 注意力模块计算时输入特征尺寸。
      fp16 (bool): 是否使用混合精度计算，默认为 False。
      activate_fun (str): 激活函数类型，可选 "LeakyReLU" 或 "ReLU"。
      patched_size (int): 注意力模块中 Patch 分割尺寸，默认为 14。
    """
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, need_attention=False, attn_size=128,
                 fp16=False, activate_fun="LeakyReLU", patched_size=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        # 卷积层，padding 使用'same'
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        # 添加注意力模块
        if need_attention:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=fp16, patched_size=patched_size))
        # 添加归一化层
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        # 根据 activate_fun 添加激活层
        if activate_fun == "LeakyReLU":
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Downsample_Vit(nn.Module):
    """
    下采样模块，融合局部分支（head、hand、global）的视觉转换。
    
    参数:
      in_channels (int): 输入通道数。
      out_channels (int): 输出通道数。
      size (int): 卷积核大小，用于初始线性转换。
      apply_batchnorm (bool): 是否采用归一化层。
      use_residual (bool): 是否采用残差连接。
      need_attention (bool): 是否在卷积中添加注意力模块。
      attn_size (int): 注意力模块输入特征尺寸（同时作为 PatchEmbed 输入尺寸）。
      fp16 (bool): 是否采用混合精度计算。
      head (int): 注意力头数。
      patch_size (int): 用于 PatchEmbed 的 Patch 尺寸。
      embed_dim (int): PatchEmbed 的投影维数基础（实际维数为 splitted_in_channel * patch_size^2）。
      patch_norm (bool): 是否对 PatchEmbed 进行归一化。
      window_size (int): Swin Transformer 窗口尺寸。
      mlp_ratio (float): MLP 模块的扩展倍率。
      qkv_bias (bool): 是否添加 qkv 的偏置。
    """
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, use_residual=True, need_attention=False,
                 attn_size=128, fp16=False, head=4, patch_size=4, embed_dim=96, patch_norm=True, window_size=7,
                 mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        self.patch_norm = patch_norm

        # 采用 LayerNorm 作为规范层
        norm_layer = nn.LayerNorm
        # 将输入通道均分为三个分支
        self.splitted_in_channel = in_channels // 3
        self.splitted_out_channel = out_channels // 3

        # PatchEmbed 提取局部补丁特征（所有分支共用同一转换）
        self.patch_embed = PatchEmbed(img_size=attn_size, patch_size=patch_size,
                                      in_chans=self.splitted_in_channel,
                                      embed_dim=self.splitted_in_channel * (patch_size ** 2),
                                      norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 构建三个 SwinTransformer 块（分别处理 head、hand、global 分支）
        self.SwinTransformerBlock_head = SwinTransformerBlock(
            dim=int(self.splitted_in_channel * (patch_size ** 2)),
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            num_heads=head, window_size=window_size,
            shift_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=0, attn_drop=0, norm_layer=norm_layer)
        
        self.SwinTransformerBlock_hand = SwinTransformerBlock(
            dim=int(self.splitted_in_channel * (patch_size ** 2)),
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            num_heads=head, window_size=window_size,
            shift_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=0, attn_drop=0, norm_layer=norm_layer)
        
        self.SwinTransformerBlock_global = SwinTransformerBlock(
            dim=int(self.splitted_in_channel * (patch_size ** 2)),
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            num_heads=head, window_size=window_size,
            shift_size=window_size // 2, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=0, attn_drop=0, norm_layer=norm_layer)

        # 初始线性变换——为各分支分别构建卷积转换层
        self.conv_linear_head = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same', bias=False)
        self.conv_linear_hand = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same', bias=False)
        self.conv_linear_global = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same', bias=False)

        # 分别针对各分支定义后续卷积与残差连接，采用两层 Attn_Conv2d 模块叠加
        self.after_vit_conv2d_head = nn.Sequential(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=need_attention, attn_size=attn_size, fp16=fp16),
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=False, attn_size=attn_size, fp16=fp16))
        self.residual_head = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel))
        
        self.after_vit_conv2d_hand = nn.Sequential(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=need_attention, attn_size=attn_size, fp16=fp16),
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=False, attn_size=attn_size, fp16=fp16))
        self.residual_hand = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel))
        
        self.after_vit_conv2d_global = nn.Sequential(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=need_attention, attn_size=attn_size, fp16=fp16),
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=False, attn_size=attn_size, fp16=fp16))
        self.residual_global = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel))
        
        # 融合三个分支后的最终卷积及下采样
        self.output_conv2d_layer = Attn_Conv2d(in_channels=out_channels, out_channels=out_channels, size=size,
                                               apply_batchnorm=apply_batchnorm, need_attention=False,
                                               attn_size=attn_size, fp16=fp16)
        self.down = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        # 分支1：head
        x_head = self.conv_linear_head(x)
        x_head = self.patch_embed(x_head)
        x_head = self.SwinTransformerBlock_head(x_head)
        x_head, _ = unpatch2(unflatten(x_head, self.splitted_in_channel), self.splitted_in_channel)
        x_head_result = self.after_vit_conv2d_head(x_head)
        x_head_res = self.residual_head(x_head)
        x_head = x_head_result + x_head_res

        # 分支2：hand
        x_hand = self.conv_linear_hand(x)
        x_hand = self.patch_embed(x_hand)
        x_hand = self.SwinTransformerBlock_hand(x_hand)
        x_hand, _ = unpatch2(unflatten(x_hand, self.splitted_in_channel), self.splitted_in_channel)
        x_hand_result = self.after_vit_conv2d_hand(x_hand)
        x_hand_res = self.residual_hand(x_hand)
        x_hand = x_hand_result + x_hand_res

        # 分支3：global
        x_global = self.conv_linear_global(x)
        x_global = self.patch_embed(x_global)
        x_global = self.SwinTransformerBlock_global(x_global)
        x_global, _ = unpatch2(unflatten(x_global, self.splitted_in_channel), self.splitted_in_channel)
        x_global_result = self.after_vit_conv2d_global(x_global)
        x_global_res = self.residual_global(x_global)
        x_global = x_global_result + x_global_res

        # 通道维度拼接三个分支结果
        x_cat = torch.concat([x_head, x_hand, x_global], dim=1)
        x_no_down = self.output_conv2d_layer(x_cat)
        x_down = self.down(x_no_down)
        return x_down, x_no_down


class Downsample_normal(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, use_residual=True, need_attention=0,
                 attn_size=128, fp16=False, head=4):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 2:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 1:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 0:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        self.result = nn.Sequential(*layers)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

        self.down = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        # if need_attention:
        #     self.attn = nn.Sequential(
        #         AttnBlock(out_channels,attn_size)
        #     )

    def forward(self, x):
        x_o = self.result(x)
        if self.use_residual:
            x_r = self.residual(x)
            x = torch.add(x_o, x_r)
        else:
            x = x_o
        x_no_down = x
        x = self.down(x)
        # if self.need_attention:
        #     x = self.attn(x)
        return x, x_no_down


class upsample(nn.Module):
    """
    上采样模块，通过双三次插值、卷积和可选的映射网络进行上采样，
    同时支持残差连接和注意力模块。

    参数:
      in_channels (int): 输入通道数。
      out_channels (int): 输出通道数。
      size (int): 卷积核大小，用于各层卷积。
      z_dim (int): 映射网络输入维数（默认512)。
      apply_batchnorm (bool): 是否采用归一化层。
      apply_dropout (bool): 是否采用dropout（预留）。
      use_residual (bool): 是否采用残差连接。
      need_attention (int): 注意力模块层数（0~3）。
      use_mapping (bool): 是否使用映射模块（即调制卷积）。
      use_fp16 (bool): 是否在映射模块中采用混合精度计算。
      attn_size (int): 注意力模块输入特征尺寸。
      atten_fp16 (bool): 注意力模块中是否采用混合精度计算。
      head (int): 注意力头数（预留）。
    """
    def __init__(self, in_channels, out_channels, size, z_dim=512, apply_batchnorm=True, apply_dropout=False,
                 use_residual=True, need_attention=0, use_mapping=False, use_fp16=False, attn_size=128,
                 atten_fp16=False, head=4):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        self.use_mapping = use_mapping
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel = 3  # 固定卷积核用于调制卷积

        # 构建上采样卷积分支
        layers = []
        # 第1层：从 in_channels 到 out_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 0:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        # 第2层
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 1:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        # 第3层
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 2:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.result = nn.Sequential(*layers)

        # 残差分支
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels)
        )
        # 上采样采用bicubic插值
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        # 用于映射调制的权重及偏置（用于调制卷积）
        self.weight = nn.Parameter(torch.randn([self.in_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = nn.Parameter(torch.zeros([self.out_channels]))
        self.affine = nn.Sequential(FullyConnectedLayer(z_dim, in_channels, bias_init=1))

    def forward(self, inputs):
        """
        参数:
          inputs: 一个元组 (x, keypoints_info)
            x: 主输入张量
            keypoints_info: 用于映射网络的关键点信息
        """
        x, keypoints_info = inputs[0], inputs[1]
        # 若使用映射模块，则通过affine网络调制卷积
        if self.use_mapping:
            styles = self.affine(keypoints_info)
            dtype = torch.float16 if (self.use_fp16 and x.device.type == 'cuda') else torch.float32
            x_o = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
                                   padding=self.conv_kernel - 1, demodulate=True, input_gain=None)
        else:
            x_o = x
        # 上采样
        x_u = self.up(x_o)
        # 主分支：卷积、激活、注意力等
        x_o = self.result(x_u)
        # 残差分支
        if self.use_residual:
            x_r = self.residual(x_u)
            x = torch.add(x_o, x_r)
        else:
            x = x_o
        return x

@misc.profiled_function
def modulated_conv2d(x, w, s, demodulate=True, padding=0, input_gain=None):
    with misc.suppress_tracer_warnings():
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    misc.assert_shape(s, [batch_size, in_channels])
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()
    w = w.unsqueeze(0)
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    """
    全连接层模块。

    参数:
        in_features (int): 输入特征的维度。
        out_features (int): 输出特征的维度。
        activation (str): 激活函数类型，默认为 'linear'，也可设置为其他激活函数名称。
        bias (bool): 是否使用偏置，默认为 True。
        lr_multiplier (float): 学习率倍率，用于调节权重和偏置的初始化规模，默认为 1.0。
        weight_init (float): 权重初始化的比例因子，默认为 1.0。
        bias_init (float or array-like): 偏置初始化值，默认为 0.
    """
    def __init__(self, in_features: int, out_features: int, activation: str = 'linear', bias: bool = True,
                 lr_multiplier: float = 1.0, weight_init: float = 1.0, bias_init=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # 初始化权重并缩放：
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (weight_init / lr_multiplier))
        if bias:
            # 若 bias_init 为标量，通过broadcast扩展为向量
            bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
            self.bias = nn.Parameter(torch.from_numpy(bias_init / lr_multiplier))
        else:
            self.bias = None

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: Tensor) -> Tensor:
        # 将权重和偏置转换至与输入相同的 dtype，并应用增益
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None

        # 根据激活类型选择计算方式
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            # 使用 bias_act 进行偏置和激活函数处理
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'activation={self.activation}')


def compute_mean_std(feats: torch.Tensor, eps=1e-8, infer=False) -> tuple[Tensor, Tensor]:
    if infer:
        n = 1
        c = 512
    else:
        n, c, _, _ = feats.shape
    feats = feats.view(n, c, -1)
    mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
    std = torch.std(feats, dim=-1).view(n, c, 1, 1) + eps
    return mean, std

class AdaIN:
    def _compute_mean_std(self, feats: torch.Tensor, eps=1e-8, infer=False) -> torch.Tensor:
        return compute_mean_std(feats, eps, infer)

    def __call__(self, content_feats: torch.Tensor, style_feats: torch.Tensor, infer: bool = False) -> torch.Tensor:
        c_mean, c_std = self._compute_mean_std(content_feats, infer=infer)
        s_mean, s_std = self._compute_mean_std(style_feats, infer=infer)
        normalized = (s_std * (content_feats - c_mean) / (c_std + 1e-8)) + s_mean
        return normalized

def process_params(part: str, down_stack):
    """
    从 down_stack 中收集指定部分（'head'、'hand' 或 'global'）的参数，并设置 requires_grad=True。
    假设每个 Downsample_Vit 模块的 get_moe_param() 返回 (head_params, hand_params, global_params)
    """
    params = []
    for down in down_stack:
        if isinstance(down, Downsample_Vit):
            head_param, hand_param, global_param = down.get_moe_param()
            if part == 'head':
                part_params = head_param
            elif part == 'hand':
                part_params = hand_param
            elif part == 'global':
                part_params = global_param
            else:
                raise ValueError(f"未知部分: {part}")
            for p in part_params:
                p.requires_grad = True
                # 假设 p 为单个参数，若 p 是可迭代对象则需要调整为 extend
                params.append(p)
    return params


def create_optimizer_and_scheduler(params, lr, betas, gamma):
    optimizer = AdaBelief(params, lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    return optimizer, scheduler


class Encoder(nn.Module):
    def __init__(self, scale=1):
        super(Encoder, self).__init__()
        self.init_content_linear_conv2d = nn.Conv2d(3, 36 // scale, kernel_size=1, padding='same', bias=False)
        self.init_style_linear_conv2d = nn.Conv2d(3, 36 // scale, kernel_size=1, padding='same', bias=False)
        self.down_1 = Downsample_Vit(36 // scale, 72 // scale, 4, apply_batchnorm=False, use_residual=False, need_attention=True, attn_size=448, fp16=True, head=1)
        self.down_2 = Downsample_Vit(72 // scale, 144 // scale, 4, need_attention=True, attn_size=224, fp16=True, head=3)
        self.down_3 = Downsample_Vit(144 // scale, 288 // scale, 4, need_attention=True, attn_size=112, fp16=True, head=2)
        self.down_4 = Downsample_Vit(288 // scale, 576 // scale, 4, need_attention=True, attn_size=56, fp16=True, head=4)
        self.down_5 = Downsample_Vit(576 // scale, 864 // scale, 4, need_attention=True, attn_size=28, fp16=True, head=4)
        self.down_6 = Downsample_normal(864 // scale, 1152 // scale, 4, need_attention=True, attn_size=14, fp16=True, head=7)
        self.down_7 = Downsample_normal(1152 // scale, 1728 // scale, 4, need_attention=3, attn_size=7, fp16=True, head=7)
        self.down_8 = Downsample_normal(1728 // scale, 2304 // scale, 4, need_attention=3, attn_size=3, fp16=True, head=9)
        self.down_stack = [self.down_1, self.down_2, self.down_3, self.down_4, self.down_5, self.down_6, self.down_7, self.down_8]

        self.ada_in = AdaIN()
        self.down_Adain = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

    def init_part_optimizer(self, lr=2e-6, scheduler_gamma=0.97):
        parts = ['head', 'hand', 'global']
        results = []
        for part in parts:
            params = process_params(part, self.down_stack)
            # 可选：检查参数是否为叶子节点
            for p in params:
                if not p.is_leaf:
                    print(f"{part} param {p} requires_grad: {p.requires_grad} is not leaf!")
            optimizer, scheduler = create_optimizer_and_scheduler(params, lr, (0.5, 0.999), scheduler_gamma)
            results.extend([optimizer, scheduler])
        return tuple(results)

    def forward(self, inputs):
        x = inputs[0]
        x_s = inputs[1]
        x = self.init_content_linear_conv2d(x)
        x_s = self.init_style_linear_conv2d(x_s)
        skips = []
        for index in range(len(self.down_stack)):
            x, x_no_down = self.down_stack[index](x)
            x_s, x_no_down_s = self.down_stack[index](x_s)
            print(f"x {index}:", str(torch.isnan(x).any()))
            print(f"x_s {index}:", str(torch.isnan(x_s).any()))
            x_skip = self.ada_in(x_no_down, x_no_down_s, False)
            x_skip = self.down_Adain(x_skip)
            print(f"x_skip {index}:", str(torch.isnan(x_skip).any()))
            skips.append(x_skip)
        x = skips[-1]
        return x, skips
