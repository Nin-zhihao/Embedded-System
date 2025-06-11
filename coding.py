import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from torch import Tensor
from torch.cuda.amp import autocast

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from multiprocessing import Pool
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import enum
import itertools

OUTPUT_CHANNELS = 3

from flash_attn.modules import mha

device = "cuda"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(attn.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        # B, N, embed_dim = x.shape
        # # 假设输入图像高宽均能被patch_size整除，此时 N = (H/patch_size) * (W/patch_size)
        # H = int(math.sqrt(N))
        # W = H  # 如果非正方形需另外传入 H, W
        # x = x.transpose(1, 2).reshape(B, embed_dim, H, W)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class AttnBlock6_b(nn.Module):
    def __init__(self, in_channels, size, min_head=4, fp16=False, patched_size=14, force_bmm_batch=False, test=False,
                 use_MultiHeadSelfAttention=True, svd=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.n = int(size // patched_size * 2)
        self.big_n = int(size // patched_size * 4)
        self.size = size
        self.force_bmm_batch = force_bmm_batch
        self.test = test
        self.use_MultiHeadSelfAttention = use_MultiHeadSelfAttention
        self.svd = svd
        # self.max_n=size//self.max_size

        if self.n > 1:
            self.need_pool = True
            head = self.n ** 2
            self.head = head
            self.use_MultiHeadSelfAttention = False
            # if self.n < 4:
            #     head = min_head
            #     self.head = head
        else:
            self.need_pool = False
            head = min_head
            self.head = head

        # self.num_worker = 4
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.norm_2 = nn.BatchNorm2d(num_features=in_channels)

        # self.norm_out = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        if self.need_pool:
            self.q = torch.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
            self.k = torch.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
            self.v = torch.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
            # if self.big_n > 1:
            # if self.big_n > 1:
            self.gap = self.n // 4
            self.patch_size = patched_size
            self.split_head_num = self.size ** 2 // self.patch_size ** 2
            self.target_global = patched_size * 4
            self.avg_p = torch.nn.AdaptiveAvgPool2d(output_size=(self.target_global, self.target_global))
            self.up_sample = torch.nn.Upsample(size=(self.size, self.size))
            self.patch_weight = 0.75
            self.global_weight = 1 - self.patch_weight

            self.proj_out = torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0, bias=False)
        if not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                self.small_attention = mha.MHA(embed_dim=size ** 2, num_heads=min_head, device='cuda',
                                               use_flash_attn=False, dwconv=True)
            else:
                self.q_group = [torch.nn.Conv2d(in_channels,
                                                in_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda()
                                for i
                                in range(head)]

                self.k_group = [torch.nn.Conv2d(in_channels,
                                                in_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda()
                                for i
                                in range(head)]
                self.v_group = [torch.nn.Conv2d(in_channels,
                                                in_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda()
                                for i
                                in range(head)]

                self.small_proj_out = torch.nn.Conv3d(head,
                                                      1,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding="same", bias=False)

    def cal_bmm_patch(self, q, k, v, b, c):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        q = q.view(b, c * self.split_head_num, self.patch_size ** 2)
        q = q.permute(0, 2, 1)  # b,hw,c

        k = k.view(b, c * self.split_head_num, self.patch_size ** 2)  # b,c,hw

        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print(w_in.shape)
        w_in = w_in * (int(c * self.split_head_num) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)

        v = v.view(b, c * self.split_head_num, self.patch_size ** 2)

        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_in = h_in.view((b, c, self.size, self.size))

        return h_in

    def cal_bmm_global(self, q, k, v, b, c):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        q = q.view(b, c, self.target_global ** 2)
        q = q.permute(0, 2, 1)  # b,hw,c

        k = k.view(b, c, self.target_global ** 2)  # b,c,hw

        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_in = w_in * (int(c) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)

        v = v.view(b, c, self.target_global ** 2)

        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_in = h_in.view((b, c, self.target_global, self.target_global))
        h_in = self.up_sample(h_in)
        return h_in

    def cal_bmm(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        # if self.mixed_precision:
        #     q = q.to(torch.float32)
        #     k = k.to(torch.float32)
        #     v = v.to(torch.float32)

        # print("q:",q.dtype)

        q = q.reshape(b * d_1, c, h_1 * w_1)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * d_1, c, h_1 * w_1)  # b,c,hw
        # q.to(torch.float8_e4m3fn)
        # k.to(torch.float8_e4m3fn)
        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print("w_in:",w_in.dtype)
        # w_in=w_in.to(torch.float32)
        w_in = w_in * (int(c) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)
        # w_in.to(torch.float8_e4m3fn)

        # attend to values
        v = v.reshape(b * d_1, c, h_1 * w_1)
        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # v.to(torch.float8_e4m3fn)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_in = h_in.reshape(b, d_1, c, h_1, w_1)
        # print("bmm h_in:", h_in.dtype)
        # print("bmm h_in:", h_in.shape)
        return h_in

    def call_funtion(self, x):
        h_ = x

        # compute attention
        b, c, h, w = h_.shape
        head = self.head
        # print("head:",head)
        d_1 = head
        if self.need_pool:
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)
            # print("q:",q.shape)

            # if self.big_n > 1:
            # h_1 = h // self.n
            # w_1 = w // self.n
            #     h_1 = 128
            #     w_1 = 128
            # else:
            # h_1 = h
            # w_1 = w
            # hw_1_kv = h_1 * w_1
            # if self.big_n > 1:
            #     hw_1_kv = 32 * (32 // self.gap)

            q_p = self.avg_p(q)
            k_p = self.avg_p(k)
            v_p = self.avg_p(v)


        else:
            if not self.use_MultiHeadSelfAttention:
                h_1 = h
                w_1 = w
                q = h_
                k = h_
                v = h_
            else:
                # q = self.q(h_)
                # k = self.k(h_)
                # v = self.v(h_)
                h_ = h_.reshape([b, c, h * w])
                # k_ = k.reshape([b, c, h * w])
                # v_ = v.reshape([b, c, h * w])

        q_list = []
        k_list = []
        v_list = []
        if self.need_pool:
            pass
            # if self.big_n > 1:
            # q = q.reshape(b, c, w * h)
            # k = k.reshape(b, c, w * h)
            # v = v.reshape(b, c, w * h)
            #
            # # q = self.q_dense(q)
            # # k = self.k_dense(k)
            # # k = self.k_dense(k)
            # for i in range(head):
            #     q_s = q[:, :, i::head]
            #     k_s = k[:, :, i::head]
            #     v_s = v[:, :, i::head]
            #     # print("q_s:",q_s.shape)
            #     if self.big_n > 1:
            #         k_s = self.max_p(k_s)
            #         # print("k_s:",k_s.shape)
            #         v_s = self.max_p(v_s)
            #         # print("v_s:", v_s.shape)
            #     q_list.append(q_s)
            #     k_list.append(k_s)
            #     v_list.append(v_s)
            #
            # q = torch.stack(q_list, dim=1)
            # # print("q:",q.shape)
            # k = torch.stack(k_list, dim=1)
            # # print("k:",k.shape)
            # v = torch.stack(v_list, dim=1)


        elif not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                pass
            else:
                q_list = []
                k_list = []
                v_list = []
                for i in range(head):
                    q_in = self.q_group[i](q)
                    k_in = self.k_group[i](k)
                    v_in = self.v_group[i](v)
                    q_list.append(q_in)
                    k_list.append(k_in)
                    v_list.append(v_in)
                q = torch.stack(q_list, dim=2)
                k = torch.stack(k_list, dim=2)
                v = torch.stack(v_list, dim=2)
        # else:
        #     q = q.view(q.size(0), -1)
        #     q -= q.min(1, keepdim=True)[0]
        #     q /= q.max(1, keepdim=True)[0]
        #     # q = q.view(batch_size, height, width)
        #
        #     k = k.view(k.size(0), -1)
        #     k -= k.min(1, keepdim=True)[0]
        #     k /= k.max(1, keepdim=True)[0]

        # if not self.need_pool:
        #     q = self.small_q(q)
        #     k = self.small_k(k)
        #     v = self.small_v(v)
        # print("after q:",q.shape)

        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            if self.need_pool:
                # if self.big_n > 0 and not self.force_bmm_batch:
                # print(h_)
                #     if self.svd:
                #         h_in = self.cal_bmm_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                #     else:
                #         h_in = self.cal_bmm_no_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                # else:
                with autocast(dtype=torch.float):

                    h_in_patch = self.cal_bmm_patch(q, k, v, b, c)
                    # if self.size==512:
                    #     print(h_in_patch)
                    #     print(h_in_patch.shape)
                    h_in_global = self.cal_bmm_global(q_p, k_p, v_p, b, c)
                    h_in = self.patch_weight * h_in_patch + self.global_weight * h_in_global
                # if self.size == 512:
                #     print(h_in_global)
                #     print(h_in_global.shape)
                #     print(h_in)
                # if self.big_n > 1:
                #     h_in = self.cal_bmm(q, k, v, b, c, h_1, h_1, d_1, h_1, h_1)
                # else:
                #     h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)


            else:

                if not self.use_MultiHeadSelfAttention:
                    with autocast(dtype=torch.float):
                        h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                else:
                    # print("use flash attention")
                    # print(h_.shape)
                    h_in = self.small_attention(h_)

                    # h_in = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

            # h_in = h_in.to(torch.float32)
        else:
            if self.need_pool:
                # if self.big_n > 0 and not self.force_bmm_batch:
                #     if self.svd:
                #         h_in = self.cal_bmm_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                #     else:
                #         h_in = self.cal_bmm_no_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                #
                # else:
                h_in_patch = self.cal_bmm_patch(q, k, v, b, c)
                # print(h_in_patch)
                # print(h_in_patch.shape)

                h_in_global = self.cal_bmm_global(q_p, k_p, v_p, b, c)
                # print(h_in_global)
                # print(h_in_global.shape)

                h_in = self.patch_weight * h_in_patch + self.global_weight * h_in_global
                # if self.big_n > 1:
                #     h_in = self.cal_bmm(q, k, v, b, c, h_1, h_1, d_1, h_1, h_1)
                # else:
                #     h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
            else:
                if not self.use_MultiHeadSelfAttention:
                    h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                else:
                    # print("use flash attention")
                    h_in = self.small_attention(h_)
                    # h_in = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        # if self.test and self.size == int(256):
        #     print("test:")
        #     h_t_s = self.cal_bmm_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
        #     q = q.reshape(b, c, w * h)
        #     k = k.reshape(b, c, w * h)
        #     v = v.reshape(b, c, w * h)
        #     for i in range(head):
        #         q_s = q[:, :, i::head]
        #         k_s = k[:, :, i::head]
        #         v_s = v[:, :, i::head]
        #         # print("q_s:",q_s.shape)
        #         q_list.append(q_s)
        #         k_list.append(k_s)
        #         v_list.append(v_s)
        #
        #     # print(q_s.shape)
        #     q = torch.stack(q_list, dim=1)
        #     k = torch.stack(k_list, dim=1)
        #     v = torch.stack(v_list, dim=1)
        #     h_t_b = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1)
        #     loss_test = nn.L1Loss()
        #     loss_result = loss_test(h_t_s, h_t_b)
        #     print(loss_result)

        if not self.need_pool:
            # h_ = torch.mean(h_in, dim=1)
            # h_ = self.proj_out(h_)
            if self.use_MultiHeadSelfAttention:
                # print(h_in.shape)

                h_ = h_in
                h_ = h_.reshape(b, c, h, w)

                # print(h_.shape)
            else:
                h_ = self.small_proj_out(h_in)
                h_ = h_.reshape(b, c, h, w)
            return h_
        else:
            # if self.big_n>1:
            with autocast(dtype=torch.float):
                h_ = self.proj_out(h_in)
                return h_

    def forward(self, x):
        # if self.big_n > 0 and not self.force_bmm_batch:
        x_n = self.norm(x)
        # if not self.need_pool and self.use_MultiHeadSelfAttention:
        #     b, c, h, w = x_n.shape
        #     x_n = x_n.reshape([b, c, h * w])
        #     x_n = self.small_attention(x_n)
        #     x_n = x_n.reshape([b, c, h, w])
        #     h_ = x_n
        # else:
        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.bfloat16):
                h_ = self.call_funtion(x_n)
        else:
            h_ = self.call_funtion(x_n)
        # print("attn")
        # if self.big_n > 0:
        #     print("x:", x)
        #     print("x_n:", x_n)
        #     print(h_)
        return x + h_


def unflatten(flattened, num_channels):
    # Alberto: Added to reconstruct from bs, n, projection_dim -> bs, n, c, h, w
    bs, n, p = flattened.size()
    unflattened = torch.reshape(flattened,
                                (bs, n, num_channels, int(np.sqrt(p // num_channels)), int(np.sqrt(p // num_channels))))
    return unflattened


def unpatch(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch == num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack(
        [torch.cat([patch for patch in x.reshape(batch_size, elem_per_axis, elem_per_axis, ch, h, w)[i]], dim=-2) for i
         in range(batch_size)], dim=0)
    restored_images = torch.stack(
        [torch.cat([patch for patch in patches_middle[i]], dim=-1) for i in range(batch_size)], dim=0).reshape(
        batch_size, 1, ch, h * elem_per_axis, w * elem_per_axis)
    return restored_images


def unpatch2(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch == num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack(
        [torch.cat([patch for patch in x.reshape(batch_size, elem_per_axis, elem_per_axis, ch, h, w)[i]], dim=-2) for i
         in range(batch_size)], dim=0)
    # print(patches_middle)
    # print(patches_middle.shape)
    restored_images = torch.stack(
        [torch.cat([patch for patch in patches_middle[i]], dim=-1) for i in range(batch_size)], dim=0).reshape(
        batch_size, ch, h * elem_per_axis, w * elem_per_axis)
    # print(restored_images)
    # print(restored_images.shape)
    # print([torch.cat([patch for patch in patches_middle[i]], dim=-1) for i in range(batch_size)][0].shape)
    return restored_images, patches_middle


class Attn_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, need_attention=False, attn_size=128,
                 fp16=False, activate_fun="LeakyReLU", patched_size=14, head=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention:
            self.layers.append(
                AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16, patched_size=patched_size))
        if apply_batchnorm:
            self.layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        self.layers.append(nn.LeakyReLU(inplace=True) if activate_fun == "LeakyReLU" else nn.ReLU(inplace=True))
        self.Sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.Sequential(x)
        return x


class Downsample_Vit(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, drop=0, use_residual=True,
                 need_attention=False,
                 attn_size=128, fp16=False, head=4, patch_size=4, embed_dim=96, patch_norm=True, window_size=7,
                 mlp_ratio=4., qkv_bias=True, ):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        self.patch_norm = patch_norm
        norm_layer = nn.LayerNorm
        # 还需要out
        self.splitted_in_channel = in_channels // 3
        self.splitted_out_channel = out_channels // 3
        self.patch_embed_hand = PatchEmbed(
            img_size=attn_size, patch_size=patch_size, in_chans=self.splitted_in_channel,
            embed_dim=self.splitted_in_channel * patch_size ** 2,
            norm_layer=norm_layer if self.patch_norm else None, )
        self.patch_embed_head = PatchEmbed(
            img_size=attn_size, patch_size=patch_size, in_chans=self.splitted_in_channel,
            embed_dim=self.splitted_in_channel * patch_size ** 2,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_global = PatchEmbed(
            img_size=attn_size, patch_size=patch_size, in_chans=self.splitted_in_channel,
            embed_dim=self.splitted_in_channel * patch_size ** 2,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed_hand.patches_resolution
        self.patches_resolution = patches_resolution
        # print(patches_resolution)

        self.SwinTransformerBlock_head = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                              input_resolution=(patches_resolution[0],
                                                                                patches_resolution[1]),
                                                              num_heads=head, window_size=window_size,
                                                              shift_size=window_size // 2,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              drop=drop, attn_drop=0,
                                                              norm_layer=norm_layer,
                                                              )

        self.SwinTransformerBlock_hand = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                              input_resolution=(patches_resolution[0],
                                                                                patches_resolution[1]),
                                                              num_heads=head, window_size=window_size,
                                                              shift_size=window_size // 2,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              drop=drop, attn_drop=0,
                                                              norm_layer=norm_layer,

                                                              )

        self.SwinTransformerBlock_global = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                                input_resolution=(patches_resolution[0],
                                                                                  patches_resolution[1]),
                                                                num_heads=head, window_size=window_size,
                                                                shift_size=window_size // 2,
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias,
                                                                drop=drop, attn_drop=0,
                                                                norm_layer=norm_layer,
                                                                )

        self.conv_linear_head = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same',
                                          bias=False)
        self.conv_linear_hand = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same',
                                          bias=False)
        self.conv_linear_global = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same',
                                            bias=False)

        self.after_vit_conv2d_hand_layers = []
        self.after_vit_conv2d_hand_layers.append(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=need_attention,
                        attn_size=attn_size, fp16=fp16, head=head))
        self.after_vit_conv2d_hand_layers.append(
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=False,
                        attn_size=attn_size, fp16=fp16, head=head))
        self.after_vit_conv2d_hand = nn.Sequential(*self.after_vit_conv2d_hand_layers)
        self.residual_hand = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel),
        )

        self.after_vit_conv2d_head_layers = []
        self.after_vit_conv2d_head_layers.append(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=need_attention,
                        attn_size=attn_size, fp16=fp16, head=head))
        self.after_vit_conv2d_head_layers.append(
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=False,
                        attn_size=attn_size, fp16=fp16, head=head))
        self.after_vit_conv2d_head = nn.Sequential(*self.after_vit_conv2d_head_layers)
        self.residual_head = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel),
        )
        self.after_vit_conv2d_global_layers = []
        self.after_vit_conv2d_global_layers.append(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=need_attention,
                        attn_size=attn_size, fp16=fp16, head=head))
        self.after_vit_conv2d_global_layers.append(
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=False,
                        attn_size=attn_size, fp16=fp16, head=head))
        self.after_vit_conv2d_global = nn.Sequential(*self.after_vit_conv2d_global_layers)
        self.residual_global = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, out_channels=self.splitted_out_channel, kernel_size=1, padding='same',
                      bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel),
        )

        self.output_conv2d_layer = Attn_Conv2d(in_channels=out_channels, out_channels=out_channels, size=size,
                                               apply_batchnorm=apply_batchnorm,
                                               need_attention=False,
                                               attn_size=attn_size, fp16=fp16, head=head)

        self.down = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

    def get_moe_param(self):
        '''
        Returns the parameters of the model in a format suitable for MoE training.
        return:
        head_param, hand_param, global_param
        '''
        head_param = []
        head_param.append(self.patch_embed_head.parameters())
        head_param.append(self.SwinTransformerBlock_head.parameters())
        head_param.append(self.after_vit_conv2d_head.parameters())
        head_param.append(self.residual_head.parameters())
        head_param = itertools.chain(*head_param)
        hand_param = []
        hand_param.append(self.patch_embed_hand.parameters())
        hand_param.append(self.SwinTransformerBlock_hand.parameters())
        hand_param.append(self.after_vit_conv2d_hand.parameters())
        hand_param.append(self.residual_hand.parameters())
        hand_param = itertools.chain(*hand_param)

        global_param = []
        global_param.append(self.patch_embed_global.parameters())
        global_param.append(self.SwinTransformerBlock_global.parameters())
        global_param.append(self.after_vit_conv2d_global.parameters())
        global_param.append(self.residual_global.parameters())
        global_param = itertools.chain(*global_param)
        return head_param, hand_param, global_param

    def forward(self, x):
        x_head = self.conv_linear_head(x)
        x_hand = self.conv_linear_hand(x)
        x_global = self.conv_linear_global(x)
        x_head = self.patch_embed_head(x_head)
        x_hand = self.patch_embed_hand(x_hand)
        x_global = self.patch_embed_global(x_global)
        x_head = self.SwinTransformerBlock_head(x_head)
        x_hand = self.SwinTransformerBlock_hand(x_hand)
        x_global = self.SwinTransformerBlock_global(x_global)

        x_head, _ = unpatch2(unflatten(x_head, self.splitted_in_channel), self.splitted_in_channel)
        x_hand, _ = unpatch2(unflatten(x_hand, self.splitted_in_channel), self.splitted_in_channel)
        x_global, _ = unpatch2(unflatten(x_global, self.splitted_in_channel), self.splitted_in_channel)
        # print(x_head.shape)
        x_head_result = self.after_vit_conv2d_head(x_head)
        x_hand_result = self.after_vit_conv2d_hand(x_head)
        x_global_result = self.after_vit_conv2d_global(x_head)
        x_head = self.residual_head(x_head)
        x_hand = self.residual_hand(x_hand)
        x_global = self.residual_global(x_global)
        x_head = x_head_result + x_head
        x_hand = x_hand_result + x_hand
        x_global = x_global_result + x_global

        x = torch.concat([x_head, x_hand, x_global], dim=1)

        x_no_down = self.output_conv2d_layer(x)

        x = self.down(x_no_down)

        return x, x_no_down


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


def compute_mean_std(
        feats: torch.Tensor, eps=1e-8, infer=False
) -> tuple[Tensor, Tensor]:
    assert (
            len(feats.shape) == 4
    ), "feature map should be 4-dimensional of the form N,C,H,W!"
    #  * Doing this to support ONNX.js inference.
    if infer:
        n = 1
        c = 512  # * fixed for vgg19
    else:
        n, c, _, _ = feats.shape

    feats = feats.view(n, c, -1)
    mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
    # print(feats.shape)
    # print(f"feats :", str(torch.isnan(feats).any()))

    std = torch.std(feats, dim=-1).view(n, c, 1, 1) + eps

    return mean, std


class AdaIN:
    """
    Adaptive Instance Normalization as proposed in
    'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
    """

    def _compute_mean_std(
            self, feats: torch.Tensor, eps=1e-8, infer=False
    ) -> torch.Tensor:
        return compute_mean_std(feats, eps, infer)

    def __call__(
            self,
            content_feats: torch.Tensor,
            style_feats: torch.Tensor,
            infer: bool = False,
    ) -> torch.Tensor:
        """
        __call__ Adaptive Instance Normalization as proposaed in
        'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.

        Args:
            content_feats (torch.Tensor): Content features
            style_feats (torch.Tensor): Style Features

        Returns:
            torch.Tensor: [description]
        """
        c_mean, c_std = self._compute_mean_std(content_feats, infer=infer)
        s_mean, s_std = self._compute_mean_std(style_feats, infer=infer)

        normalized = (s_std * (content_feats - c_mean) / (c_std + 1e-8)) + s_mean

        return normalized


class Upsample_Vit(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, size, z_dim=512, apply_batchnorm=True, use_residual=True,
                 need_attention=False,
                 attn_size=128, use_mapping=True, fp16=False, map_use_fp16=False, head=4, patch_size=4, embed_dim=96,
                 patch_norm=True,
                 window_size=7, drop=0.0,
                 mlp_ratio=4., qkv_bias=True, up_sample_size=None, need_up_sample=True,final_layer = False):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        self.use_mapping = use_mapping
        self.patch_norm = patch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.map_use_fp16 = map_use_fp16
        self.need_up_sample = need_up_sample
        self.final_layer = final_layer
        self.conv_kernel = 3
        norm_layer = nn.LayerNorm
        # 还需要out
        self.splitted_in_channel = in_channels // 3
        self.splitted_out_channel = out_channels // 3
        self.patch_embed_hand = PatchEmbed(
            img_size=attn_size, patch_size=patch_size, in_chans=self.splitted_in_channel,
            embed_dim=self.splitted_in_channel * patch_size ** 2,
            norm_layer=norm_layer if self.patch_norm else None, )
        self.patch_embed_head = PatchEmbed(
            img_size=attn_size, patch_size=patch_size, in_chans=self.splitted_in_channel,
            embed_dim=self.splitted_in_channel * patch_size ** 2,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_global = PatchEmbed(
            img_size=attn_size, patch_size=patch_size, in_chans=self.splitted_in_channel,
            embed_dim=self.splitted_in_channel * patch_size ** 2,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed_hand.patches_resolution
        self.patches_resolution = patches_resolution
        # print(patches_resolution)

        self.SwinTransformerBlock_head = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                              input_resolution=(patches_resolution[0],
                                                                                patches_resolution[1]),
                                                              num_heads=head, window_size=window_size,
                                                              shift_size=window_size // 2,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              drop=drop, attn_drop=0,
                                                              norm_layer=norm_layer,
                                                              )

        self.SwinTransformerBlock_hand = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                              input_resolution=(patches_resolution[0],
                                                                                patches_resolution[1]),
                                                              num_heads=head, window_size=window_size,
                                                              shift_size=window_size // 2,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              drop=drop, attn_drop=0,
                                                              norm_layer=norm_layer,
                                                              )

        self.SwinTransformerBlock_global = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                                input_resolution=(patches_resolution[0],
                                                                                  patches_resolution[1]),
                                                                num_heads=head, window_size=window_size,
                                                                shift_size=window_size // 2,
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias,
                                                                drop=drop, attn_drop=0,
                                                                norm_layer=norm_layer,
                                                                )

        self.conv_linear_head = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same',
                                          bias=False)
        self.conv_linear_hand = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same',
                                          bias=False)
        self.conv_linear_global = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same',
                                            bias=False)

        self.after_vit_conv2d_hand_layers = []
        self.after_vit_conv2d_hand_layers.append(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=need_attention,
                        attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu"))
        self.after_vit_conv2d_hand_layers.append(
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=False,
                        attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu"))
        self.after_vit_conv2d_hand = nn.Sequential(*self.after_vit_conv2d_hand_layers)
        self.residual_hand = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel),
        )

        self.after_vit_conv2d_head_layers = []
        self.after_vit_conv2d_head_layers.append(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=need_attention,
                        attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu"))
        self.after_vit_conv2d_head_layers.append(
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=False,
                        attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu"))
        self.after_vit_conv2d_head = nn.Sequential(*self.after_vit_conv2d_head_layers)
        self.residual_head = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel),
        )
        self.after_vit_conv2d_global_layers = []
        self.after_vit_conv2d_global_layers.append(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=need_attention,
                        attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu"))
        self.after_vit_conv2d_global_layers.append(
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm,
                        need_attention=False,
                        attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu"))
        self.after_vit_conv2d_global = nn.Sequential(*self.after_vit_conv2d_global_layers)
        self.residual_global = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, out_channels=self.splitted_out_channel, kernel_size=1, padding='same',
                      bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel),
        )

        self.output_conv2d_layer = Attn_Conv2d(in_channels=out_channels, out_channels=out_channels, size=size,
                                               apply_batchnorm=apply_batchnorm,
                                               need_attention=False,
                                               attn_size=attn_size, fp16=fp16, head=head, activate_fun="relu")
        if need_up_sample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(None if up_sample_size is not None else 2),
                            size=(up_sample_size if up_sample_size is not None else None),
                            mode='bicubic',
                            align_corners=True)
            )

        self.weight = torch.nn.Parameter(
            torch.randn([self.in_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.affine = nn.Sequential(FullyConnectedLayer(z_dim, in_channels, bias_init=1))

    def get_moe_param(self):
        '''
        Returns the parameters of the model in a format suitable for MoE training.
        return:
        head_param, hand_param, global_param
        '''
        head_param = []
        head_param.append(self.patch_embed_head.parameters())
        head_param.append(self.SwinTransformerBlock_head.parameters())
        head_param.append(self.after_vit_conv2d_head.parameters())
        head_param.append(self.residual_head.parameters())
        head_param = itertools.chain(*head_param)
        hand_param = []
        hand_param.append(self.patch_embed_hand.parameters())
        hand_param.append(self.SwinTransformerBlock_hand.parameters())
        hand_param.append(self.after_vit_conv2d_hand.parameters())
        hand_param.append(self.residual_hand.parameters())
        hand_param = itertools.chain(*hand_param)

        global_param = []
        global_param.append(self.patch_embed_global.parameters())
        global_param.append(self.SwinTransformerBlock_global.parameters())
        global_param.append(self.after_vit_conv2d_global.parameters())
        global_param.append(self.residual_global.parameters())
        global_param = itertools.chain(*global_param)
        return head_param, hand_param, global_param

    def forward(self, inputs):
        if not self.final_layer:
            x, keypoints_info = inputs[0], inputs[1]
        else:
            x = inputs[0]
            keypoints_info = None
        print(x.shape)

        if self.use_mapping:
            styles = self.affine(keypoints_info)
            dtype = torch.float16 if (self.map_use_fp16 and x.device.type == 'cuda') else torch.float32
            x_o = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
                                   padding="same", demodulate=True, input_gain=None)
            print(x_o.shape)
        else:
            x_o = x.unsqueeze(0)
        if self.need_up_sample:
            x_u = self.up(x_o)
        else:
            x_u = x_o
        print(x_u.shape)
        x_head = self.conv_linear_head(x_u)
        x_hand = self.conv_linear_hand(x_u)
        x_global = self.conv_linear_global(x_u)
        x_head = self.patch_embed_head(x_head)
        x_hand = self.patch_embed_hand(x_hand)
        x_global = self.patch_embed_global(x_global)
        x_head = self.SwinTransformerBlock_head(x_head)
        x_hand = self.SwinTransformerBlock_hand(x_hand)
        x_global = self.SwinTransformerBlock_global(x_global)

        x_head, _ = unpatch2(unflatten(x_head, self.splitted_in_channel), self.splitted_in_channel)
        x_hand, _ = unpatch2(unflatten(x_hand, self.splitted_in_channel), self.splitted_in_channel)
        x_global, _ = unpatch2(unflatten(x_global, self.splitted_in_channel), self.splitted_in_channel)
        # print(x_head.shape)
        x_head_result = self.after_vit_conv2d_head(x_head)
        x_hand_result = self.after_vit_conv2d_hand(x_head)
        x_global_result = self.after_vit_conv2d_global(x_head)
        x_head = self.residual_head(x_head)
        x_hand = self.residual_hand(x_hand)
        x_global = self.residual_global(x_global)
        x_head = x_head_result + x_head
        x_hand = x_hand_result + x_hand
        x_global = x_global_result + x_global

        x = torch.concat([x_head, x_hand, x_global], dim=1)

        x = self.output_conv2d_layer(x)

        return x


class Upsample_Normal(nn.Module):

    def __init__(self, in_channels, out_channels, size, z_dim=512, apply_batchnorm=True, drop=0.,
                 use_residual=True,
                 need_attention=0, use_mapping=False, map_use_fp16=False, attn_size=128, fp16=False,
                 up_sample_size=None, head=4):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        self.use_mapping = use_mapping
        self.map_use_fp16 = map_use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel = 3

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 0:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 1:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 2:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=head, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        if drop > 0:
            layers.append(nn.Dropout(p=drop))

        self.result = nn.Sequential(*layers)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(None if up_sample_size is not None else 2),
                        size=(up_sample_size if up_sample_size is not None else None),
                        mode='bicubic',
                        align_corners=True)
        )

        self.weight = torch.nn.Parameter(
            torch.randn([self.in_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.affine = nn.Sequential(FullyConnectedLayer(z_dim, in_channels, bias_init=1))

    def forward(self, inputs):
        # print(inputs)
        x, keypoints_info = inputs[0], inputs[1]
        if self.use_mapping:
            styles = self.affine(keypoints_info)
            dtype = torch.float16 if (self.map_use_fp16 and x.device.type == 'cuda') else torch.float32
            x_o = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
                                   padding=self.conv_kernel - 1, demodulate=True, input_gain=None)
        else:
            x_o = x
        x_u = self.up(x_o)
        x_o = self.result(x_u)
        if self.use_residual:
            x_r = self.residual(x_u)
            x = torch.add(x_o, x_r)
        else:
            x = x_o

        return x


@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
        s,  # Style tensor: [batch_size, in_channels]
        demodulate=True,  # Apply weight demodulation?
        padding=0,  # Padding: int or [padH, padW]
        input_gain=None,
        # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 bias=True,  # Apply additive bias before the activation function?
                 lr_multiplier=1,  # Learning rate multiplier.
                 weight_init=1,  # Initial standard deviation of the weight tensor.
                 bias_init=0,  # Initial value of the additive bias.
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class ModelScale(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


class Encoder(nn.Module):
    def __init__(self, scale=1):
        super(Encoder, self).__init__()
        self.init_content_linear_conv2d = nn.Conv2d(3, 36 // scale, kernel_size=1, padding='same', bias=False)
        self.init_style_linear_conv2d = nn.Conv2d(3, 36 // scale, kernel_size=1, padding='same', bias=False)
        self.down_1 = Downsample_Vit(36 // scale, 72 // scale, 4, apply_batchnorm=False, use_residual=False,
                                     need_attention=True, attn_size=448, fp16=True, head=1)  # 448 -> 224
        self.down_2 = Downsample_Vit(72 // scale, 144 // scale, 4, need_attention=True, attn_size=224, fp16=True,
                                     head=2)  # 224 -> 112
        self.down_3 = Downsample_Vit(144 // scale, 288 // scale, 4, need_attention=True, attn_size=112, fp16=True,
                                     head=2)  # 112 -> 56
        self.down_4 = Downsample_Vit(288 // scale, 576 // scale, 4, need_attention=True,
                                     attn_size=56, fp16=True, head=4)  # 56 -> 28
        self.down_5 = Downsample_Vit(576 // scale, 864 // scale, 4, need_attention=True, attn_size=28,
                                     fp16=True, head=4)  # 28 -> 14
        self.down_6 = Downsample_normal(864 // scale, 1152 // scale, 4, need_attention=True, attn_size=14,
                                        fp16=True, head=7)  # 14 -> 7
        self.down_7 = Downsample_normal(1152 // scale, 1728 // scale, 4, need_attention=3, attn_size=7,
                                        fp16=True, head=7)  # 7 -> 3
        self.down_8 = Downsample_normal(1728 // scale, 2304 // scale, 4, need_attention=3, attn_size=3,
                                        fp16=True, head=9)  # 3 -> 1

        self.down_stack = [self.down_1, self.down_2, self.down_3, self.down_4, self.down_5, self.down_6, self.down_7,
                           self.down_8]

        self.ada_in = AdaIN()
        self.down_Adain = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.head_part_optimizer = None
        self.hand_part_optimizer = None
        self.global_part_optimizer = None
        self.head_part_scheduler = None
        self.hand_part_scheduler = None
        self.global_part_scheduler = None

    def init_part_optimizer(self, lr=2e-6, scheduler_gamma=0.97):
        head_params = []
        hand_params = []
        global_params = []

        for down in self.down_stack:
            print(isinstance(down, Downsample_Vit))
            if isinstance(down, Downsample_Vit):
                head_param, hand_param, global_param = down.get_moe_param()

                head_params.append(head_param)
                hand_params.append(hand_param)
                global_params.append(global_param)

        head_params = itertools.chain(*head_params)
        hand_params = itertools.chain(*hand_params)
        global_params = itertools.chain(*global_params)
        head_part_optimizer = AdaBelief(head_params, lr=lr, betas=(0.5, 0.999))
        head_part_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=head_part_optimizer,
                                                                     gamma=scheduler_gamma)

        hand_part_optimizer = AdaBelief(hand_params, lr=lr, betas=(0.5, 0.999))
        hand_part_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hand_part_optimizer,
                                                                     gamma=scheduler_gamma)

        global_part_optimizer = AdaBelief(global_params, lr=lr, betas=(0.5, 0.999))
        global_part_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=global_part_optimizer,
                                                                       gamma=scheduler_gamma)

        return head_part_optimizer, head_part_scheduler, hand_part_optimizer, hand_part_scheduler, global_part_optimizer, global_part_scheduler

    def forward(self, inputs):
        x = inputs[0]
        x_s = inputs[1]
        x = self.init_content_linear_conv2d(x)
        x_s = self.init_style_linear_conv2d(x_s)

        skips = []
        for index in range(len(self.down_stack)):
            # print("x:", x.shape)
            # print("x_s:", x_s.shape)
            x, x_no_down = self.down_stack[index](x)
            x_s, x_no_down_s = self.down_stack[index](x_s)
            # print(f"x {index}:",x)
            # print(f"x_s {index}:",x_s)
            # print(f"x {index}:", str(torch.isnan(x).any()))
            # print(f"x_s {index}:", str(torch.isnan(x_s).any()))

            # x_skip = adaptive_instance_normalization(x_s, x)
            x_skip = self.ada_in(x_no_down, x_no_down_s, False)
            x_skip = self.down_Adain(x_skip)
            # print(f"x_skip {index}:", x_skip)
            # print(f"x_skip {index}:", str(torch.isnan(x_skip).any()))

            # x_skip = tf.keras.layers.Concatenate()([x_s, x])
            skips.append(x_skip)
        x = skips[-1]

        return x, skips


class Decoder(nn.Module):
    def __init__(self, scale=1):
        super(Decoder, self).__init__()
        self.up_1 = Upsample_Normal(2304 // scale, 1728 // scale, 4,
                                    need_attention=True, attn_size=3, fp16=True, head=9, drop=0.2,
                                    up_sample_size=[3, 3])  # 1->3
        self.up_2 = Upsample_Normal(1728 * 2 // scale, 1152 // scale, 4,
                                    need_attention=True, attn_size=7, fp16=True, head=7, drop=0.1,
                                    up_sample_size=[7, 7])  # 3->7
        self.up_3 = Upsample_Normal(1152 * 2 // scale, 864 // scale, 4,
                                    need_attention=True, attn_size=14, fp16=True, head=7, drop=0.1)  # 7->14
        self.up_4 = Upsample_Vit(864 * 2 // scale, 576 // scale, 4,
                                 need_attention=True, attn_size=28, fp16=True, head=4, drop=0.1)  # 14->28
        self.up_5 = Upsample_Vit(576 * 2 // scale, 288 // scale, 4,
                                 need_attention=True, attn_size=56, fp16=True, head=4)  # 28->56
        self.up_6 = Upsample_Vit(288 * 2 // scale, 144 // scale, 4,
                                 need_attention=True, attn_size=112, fp16=True, head=2)  # 56->112
        self.up_7 = Upsample_Vit(144 * 2 // scale, 72 // scale, 4,
                                 need_attention=True, attn_size=224, fp16=True, head=2)  # 112->224
        self.up_8 = Upsample_Vit(72 * 2 // scale, 36 // scale, 4,
                                 apply_batchnorm=False, use_residual=False,
                                 need_attention=True, attn_size=448, fp16=True, head=1)  # 224->448

        self.up_final = Upsample_Vit(36 // scale, 18 // scale, 4,
                                     apply_batchnorm=False, use_residual=False,
                                     need_attention=True, attn_size=448, fp16=True, head=1,
                                     need_up_sample=False,use_mapping=False,final_layer=True)  # 224->448

        self.up_stack = [self.up_1, self.up_2, self.up_3, self.up_4, self.up_5, self.up_6, self.up_7, self.up_8]

        self.out_layer = nn.Sequential(
            nn.Conv2d(18 // scale, OUTPUT_CHANNELS, kernel_size=1, padding="same", bias=False),
            nn.Tanh(),
        )

        self.head_part_optimizer = None
        self.hand_part_optimizer = None
        self.global_part_optimizer = None
        self.head_part_scheduler = None
        self.hand_part_scheduler = None
        self.global_part_scheduler = None

    def init_part_optimizer(self, lr=2e-6, scheduler_gamma=0.97):
        head_params = []
        hand_params = []
        global_params = []

        for down in self.down_stack:
            print(isinstance(down, Downsample_Vit))
            if isinstance(down, Downsample_Vit):
                head_param, hand_param, global_param = down.get_moe_param()

                head_params.append(head_param)
                hand_params.append(hand_param)
                global_params.append(global_param)

        head_params = itertools.chain(*head_params)
        hand_params = itertools.chain(*hand_params)
        global_params = itertools.chain(*global_params)
        head_part_optimizer = AdaBelief(head_params, lr=lr, betas=(0.5, 0.999))
        head_part_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=head_part_optimizer,
                                                                     gamma=scheduler_gamma)

        hand_part_optimizer = AdaBelief(hand_params, lr=lr, betas=(0.5, 0.999))
        hand_part_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=hand_part_optimizer,
                                                                     gamma=scheduler_gamma)

        global_part_optimizer = AdaBelief(global_params, lr=lr, betas=(0.5, 0.999))
        global_part_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=global_part_optimizer,
                                                                       gamma=scheduler_gamma)

        return head_part_optimizer, head_part_scheduler, hand_part_optimizer, hand_part_scheduler, global_part_optimizer, global_part_scheduler

    def forward(self, all_inputs):
        skips = all_inputs[0]
        keypoints_info_f = all_inputs[1]
        x = skips[-1]
        for skip in skips:
            print(f"skip:", skip.shape)
            print(f"skip nan:", str(torch.isnan(skip).any()))

        for index in range(len(self.up_stack)):
            inputs = (x, keypoints_info_f)
            x = self.up_stack[index](inputs)
            # print(f"skips len:", len(skips))
            # print(f"up_stack len:", len(self.up_stack))
            skip_in = skips[ len(self.up_stack)- index-2]
            print(f"x {index}:",x.shape)
            print(f"skip_in {index}:", skip_in.shape)
            print(f"x nan {index}:", str(torch.isnan(x).any()))
            if index == len(self.up_stack) - 1:
                x = x
            else:
                x = torch.concat([x, skip_in], dim=1)
        x = self.up_final(x)
        x = self.out_layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, lr_multiplier=0.01, z_dim=512):
        scale = 2
        super(Generator, self).__init__()

        self.down_1 = nn.Sequential(
            Downsample_normal(3, 64 // scale, 4, apply_batchnorm=False, use_residual=False, need_attention=1,
                              attn_size=512, fp16=True, head=1))
        self.down_2 = nn.Sequential(
            Downsample_normal(64 // scale, 128 // scale, 4, need_attention=2, attn_size=256, fp16=True, head=3))
        self.down_3 = nn.Sequential(
            Downsample_normal(128 // scale, 256 // scale, 4, need_attention=2, attn_size=128, fp16=True, head=2))
        self.down_4 = nn.Sequential(
            Downsample_normal(256 // scale, 512 // scale, 4, need_attention=2,
                              attn_size=64, fp16=True, head=4))  # (batch_size, 16, 16, 512) 32
        self.down_5 = nn.Sequential(
            Downsample_normal(512 // scale, 768 // scale, 4, need_attention=3, attn_size=32,
                              fp16=True, head=4))  # (batch_size, 8, 8, 512) 16
        self.down_6 = nn.Sequential(
            Downsample_normal(768 // scale, 1024 // scale, 4, need_attention=3, attn_size=16,
                              fp16=True, head=4))  # (batch_size, 4, 4, 512) 8
        self.down_7 = nn.Sequential(
            Downsample_normal(1024 // scale, 1536 // scale, 4, need_attention=3, attn_size=8,
                              fp16=True, head=4))  # (batch_size, 2, 2, 512) 4
        self.down_8 = nn.Sequential(
            Downsample_normal(1536 // scale, 2048 // scale, 4, need_attention=3, attn_size=4,
                              fp16=True, head=4))  # (batch_size, 1, 1, 512) 2
        self.down_9 = nn.Sequential(
            Downsample_normal(2048 // scale, 4096 // scale, 2, apply_batchnorm=True, use_residual=True,
                              need_attention=3, attn_size=2, fp16=True, head=4))  # 1
        # downsample(512, 4),  # (batch_size, 1, 1, 512)

        self.down_stack = [self.down_1, self.down_2, self.down_3, self.down_4, self.down_5, self.down_6, self.down_7,
                           self.down_8, self.down_9]

        self.up_1 = nn.Sequential(
            Upsample_Normal(4096 // scale, 2048 // scale, 2, apply_dropout=True, need_attention=3, attn_size=2,
                            atten_fp16=True, head=4))  # 2
        self.up_2 = nn.Sequential(
            Upsample_Normal(4096 // scale, 1536 // scale, 4, apply_dropout=True, need_attention=3, attn_size=4,
                            atten_fp16=True, head=4))  # 4
        self.up_3 = nn.Sequential(Upsample_Normal(3072 // scale, 1024 // scale, 4, apply_dropout=True,
                                                  need_attention=3, attn_size=8,
                                                  atten_fp16=True, head=4))  # (batch_size, 2, 2, 1024) 8
        self.up_4 = nn.Sequential(
            Upsample_Normal(2048 // scale, 768 // scale, 4, need_attention=3, attn_size=16,
                            atten_fp16=True, head=4))  # (batch_size, 4, 4, 1024) 16
        self.up_5 = nn.Sequential(
            Upsample_Normal(1536 // scale, 512 // scale, 4, need_attention=3, attn_size=32,
                            atten_fp16=True, head=4))  # (batch_size, 8, 8, 1024) 32
        self.up_6 = nn.Sequential(
            Upsample_Normal(1024 // scale, 256 // scale, 4, need_attention=2,
                            attn_size=64, atten_fp16=True, head=4))  # (batch_size, 16, 16, 1024) 64
        self.up_7 = nn.Sequential(
            Upsample_Normal(512 // scale, 128 // scale, 4, need_attention=2,
                            attn_size=128, atten_fp16=True, head=2))  # (batch_size, 32, 32, 512) 128
        self.up_8 = nn.Sequential(
            Upsample_Normal(256 // scale, 64 // scale, 4, need_attention=2,
                            attn_size=256, atten_fp16=True, head=2))  # (batch_size, 64, 64, 256) 256

        # (batch_size, 128, 128, 128) 512
        self.up_stack = [self.up_1, self.up_2, self.up_3, self.up_4, self.up_5, self.up_6, self.up_7, self.up_8]

        # self.up_1_cross_attention_D = nn.Sequential(AttnBlock6_cross(2048 // scale, 2, 4, fp16=True))  # 2
        # self.up_2_cross_attention_D = nn.Sequential(AttnBlock6_cross(1536 // scale, 4, 4, fp16=True))  # 4
        # self.up_3_cross_attention_D = nn.Sequential(AttnBlock6_cross(1024 // scale, 8, 4, fp16=True))
        # self.up_4_cross_attention_D = nn.Sequential(AttnBlock6_cross(768 // scale, 16, 4, fp16=True))
        # self.up_5_cross_attention_D = nn.Sequential(AttnBlock6_cross(512 // scale, 32, 4, fp16=True))
        # self.up_6_cross_attention_D = nn.Sequential(AttnBlock6_cross(256 // scale, 64, 4, fp16=True))
        # self.up_7_cross_attention_D = nn.Sequential(AttnBlock6_cross(128 // scale, 128, 4, fp16=True))
        # self.up_8_cross_attention_D = nn.Sequential(AttnBlock6_cross(64 // scale, 256, 4, fp16=True))
        #
        # self.up_1_cross_attention_E = nn.Sequential(AttnBlock6_cross(2048 // scale, 2, 4, fp16=True))  # 2
        # self.up_2_cross_attention_E = nn.Sequential(AttnBlock6_cross(1536 // scale, 4, 4, fp16=True))  # 4
        # self.up_3_cross_attention_E = nn.Sequential(AttnBlock6_cross(1024 // scale, 8, 4, fp16=True))
        # self.up_4_cross_attention_E = nn.Sequential(AttnBlock6_cross(768 // scale, 16, 4, fp16=True))
        # self.up_5_cross_attention_E = nn.Sequential(AttnBlock6_cross(512 // scale, 32, 4, fp16=True))
        # self.up_6_cross_attention_E = nn.Sequential(AttnBlock6_cross(256 // scale, 64, 4, fp16=True))
        # self.up_7_cross_attention_E = nn.Sequential(AttnBlock6_cross(128 // scale, 128, 4, fp16=True))
        # self.up_8_cross_attention_E = nn.Sequential(AttnBlock6_cross(64 // scale, 256, 4, fp16=True))
        #
        # self.up_cross_attention_D_stack = [self.up_1_cross_attention_D, self.up_2_cross_attention_D,
        #                                    self.up_3_cross_attention_D, self.up_4_cross_attention_D,
        #                                    self.up_5_cross_attention_D, self.up_6_cross_attention_D,
        #                                    self.up_7_cross_attention_D, self.up_8_cross_attention_D]
        # self.up_cross_attention_E_stack = [self.up_1_cross_attention_E, self.up_2_cross_attention_E,
        #                                    self.up_3_cross_attention_E, self.up_4_cross_attention_E,
        #                                    self.up_5_cross_attention_E, self.up_6_cross_attention_E,
        #                                    self.up_7_cross_attention_E, self.up_8_cross_attention_E]

        self.mapping_network = nn.Sequential(FullyConnectedLayer(266, z_dim),
                                             FullyConnectedLayer(z_dim, z_dim, activation='lrelu',
                                                                 lr_multiplier=lr_multiplier),
                                             FullyConnectedLayer(z_dim, z_dim, activation='lrelu',
                                                                 lr_multiplier=lr_multiplier))
        # self.affine_stack = [nn.Sequential(FullyConnectedLayer(z_dim, 4096//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 2048//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 1536//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 1024//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 768//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 512//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 256//4, bias_init=1)),
        #                      nn.Sequential(FullyConnectedLayer(z_dim, 128//4, bias_init=1))]

        self.final_up = nn.Sequential(
            upsample(128 // scale, 32 // scale, 4, need_attention=1, attn_size=512, atten_fp16=True, head=1))
        self.out_layer = nn.Sequential(
            nn.Conv2d(32 // scale, OUTPUT_CHANNELS, kernel_size=1, padding="same", bias=False),
            nn.Tanh(),
        )

        # self.keypoint_seq_bodies = nn.Sequential(
        #     nn.Linear(46, 1024),
        #     nn.InstanceNorm1d(1024),
        # )
        # self.keypoint_input_faces = nn.Sequential(
        #     nn.Linear(136, 1024),
        #     nn.InstanceNorm1d(1024),
        # )
        # self.keypoint_seq_left_hands = nn.Sequential(
        #     nn.Linear(42, 1024),
        #     nn.InstanceNorm1d(1024),
        # )
        # self.keypoint_input_right_hands = nn.Sequential(
        #     nn.Linear(42, 1024),
        #     nn.InstanceNorm1d(1024),
        # )
        self.ada_in = AdaIN()
        self.down_Adain = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

    def forward(self, inputs_content, inputs_style, keypoint_input_bodies, keypoint_input_faces,
                keypoint_input_left_hands, keypoint_input_right_hands):
        # x = torch.concat([inputs_content, inputs_content, inputs_content], dim=1)
        # x_s = torch.concat([inputs_style, inputs_style, inputs_style, inputs_style], dim=1)
        x = inputs_content
        x_s = inputs_style
        # print(type(keypoint_input_bodies))

        bodies = nn.Flatten()(keypoint_input_bodies)
        faces = nn.Flatten()(keypoint_input_faces)
        left_hands = nn.Flatten()(keypoint_input_left_hands)
        right_hands = nn.Flatten()(keypoint_input_right_hands)

        keypoints_info = torch.concat([bodies, faces, left_hands, right_hands], dim=1)
        # print(type(keypoints_info))
        keypoints_info_f = self.mapping_network(keypoints_info)
        # print(keypoints_info_f.shape)

        # bodies = keypoint_seq_bodies(bodies)
        # faces = keypoint_seq_faces(faces)
        # left_hands = keypoint_seq_left_hands(left_hands)
        # right_hands = keypoint_seq_right_hands(right_hands)
        skips = []
        for index in range(len(self.down_stack)):
            # print("x:", x.shape)
            # print("x_s:", x_s.shape)
            x, x_no_down = self.down_stack[index](x)
            x_s, x_no_down_s = self.down_stack[index](x_s)
            # print(f"x {index}:",x)
            # print(f"x_s {index}:",x_s)
            # print(f"x {index}:", str(torch.isnan(x).any()))
            # print(f"x_s {index}:", str(torch.isnan(x_s).any()))

            # x_skip = adaptive_instance_normalization(x_s, x)
            x_skip = self.ada_in(x_no_down, x_no_down_s, False)
            x_skip = self.down_Adain(x_skip)
            # print(f"x_skip {index}:", x_skip)
            # print(f"x_skip {index}:", str(torch.isnan(x_skip).any()))

            # x_skip = tf.keras.layers.Concatenate()([x_s, x])
            skips.append(x_skip)
        x = skips[-1]
        # skips_R = reversed(skips[:-1])
        index = 0
        for index in range(len(self.up_stack)):
            # print("x:", x.shape)
            # print("index:", index)
            # print("index:", keypoints_info_f.shape)
            inputs = (x, keypoints_info_f)
            x = self.up_stack[index](inputs)
            skip_in = skips[len(self.up_stack) - 1 - index]
            # print(skip_in.shape)
            # x = self.up_cross_attention_D_stack[index]([x, skip_in])
            # skip_in = self.up_cross_attention_E_stack[index]([skip_in, x])

            x = torch.concat([x, skip_in], dim=1)
        inputs = (x, keypoints_info_f)
        x = self.final_up(inputs)
        x = self.out_layer(x)
        return x


class downsample_d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, need_down=True, need_attention=True,
                 attn_size=128, atten_fp16=False):
        super().__init__()
        self.need_attention = need_attention

        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=1, padding="same", bias=False),
        )
        if need_attention:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=2, fp16=atten_fp16, force_bmm_batch=False))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))
        if need_down:
            layers.append(nn.MaxPool2d(stride=2, kernel_size=2))
        self.result = nn.Sequential(*layers)
        # if need_attention:
        #     self.attn = nn.Sequential(
        #         AttnBlock3(out_channels, attn_size)
        #     )
        # else:
        #     self.attn = None

    def forward(self, x):
        x = self.result(x)
        # if self.need_attention:
        #     x = self.attn(x)
        return x


class Discriminator(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self):
        super().__init__()
        out_channels = 512
        self.downsample_d_1 = downsample_d(6, 128, 4, need_attention=False, attn_size=512, atten_fp16=True)
        self.downsample_d_2 = downsample_d(128, 256, 4, need_attention=True, attn_size=256, atten_fp16=True)
        self.downsample_d_3 = downsample_d(256, 384, 4, need_attention=True, attn_size=128, atten_fp16=True)
        self.downsample_d_4 = downsample_d(384, out_channels, 4, need_attention=True, attn_size=64, atten_fp16=True)
        # self.downsample_d_5 = downsample_d(512, 1024, 4)
        self.conv_out1 = nn.Sequential(
            torch.nn.ZeroPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            # AttnBlock(out_channels),
            torch.nn.ZeroPad2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, inp, tar):
        x = torch.concat([inp, tar], dim=1)
        x = self.downsample_d_1(x)
        x = self.downsample_d_2(x)
        x = self.downsample_d_3(x)
        x = self.downsample_d_4(x)
        # x = self.downsample_d_5(x)
        x = self.conv_out1(x)
        return x


class Discriminator_Hand(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self):
        super().__init__()
        out_channels = 512
        self.downsample_d_1 = downsample_d(6, 128, 4, need_attention=False, attn_size=128, atten_fp16=True)
        self.downsample_d_2 = downsample_d(128, 256, 4, need_down=False, need_attention=True, attn_size=64,
                                           atten_fp16=True)
        self.downsample_d_3 = downsample_d(256, 384, 4, need_down=False, need_attention=True, attn_size=64,
                                           atten_fp16=True)
        self.downsample_d_4 = downsample_d(384, out_channels, 4, need_attention=True, attn_size=64, atten_fp16=True)
        self.conv_out2 = nn.Sequential(
            torch.nn.ZeroPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            # AttnBlock(out_channels),
            torch.nn.ZeroPad2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, inp, tar):
        x = torch.concat([inp, tar], dim=1)
        x = self.downsample_d_1(x)
        x = self.downsample_d_2(x)
        x = self.downsample_d_3(x)
        x = self.downsample_d_4(x)
        x = self.conv_out2(x)
        return x


class Discriminator_Head(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self):
        super().__init__()
        out_channels = 512
        self.downsample_d_1 = downsample_d(6, 128, 4, need_attention=False, attn_size=128, atten_fp16=True)
        self.downsample_d_2 = downsample_d(128, 256, 4, need_down=False, need_attention=True, attn_size=64,
                                           atten_fp16=True)
        self.downsample_d_3 = downsample_d(256, 384, 4, need_down=False, need_attention=True, attn_size=64,
                                           atten_fp16=True)
        self.downsample_d_4 = downsample_d(384, out_channels, 4, need_attention=True, attn_size=64, atten_fp16=True)
        self.conv_out2 = nn.Sequential(
            torch.nn.ZeroPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            # AttnBlock(out_channels),
            torch.nn.ZeroPad2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, inp, tar):
        x = torch.concat([inp, tar], dim=1)
        x = self.downsample_d_1(x)
        x = self.downsample_d_2(x)
        x = self.downsample_d_3(x)
        x = self.downsample_d_4(x)
        x = self.conv_out2(x)
        return x


from torchinfo import summary

channel = 27
imgsize = 448
patchsize = 4
x = torch.zeros([1, channel, imgsize, imgsize]).cuda()
# patch_embed = PatchEmbed(
#     img_size=imgsize, patch_size=patchsize, in_chans=channel, embed_dim=channel * patchsize ** 2,
#     norm_layer=nn.LayerNorm).cuda()
#
# # patch_encoder = PatchEncoder(
# #             img_size=512, patch_size=4, num_channels=32).cuda()
# embed = patch_embed(x)
# print(embed.shape)
# patches_resolution = patch_embed.patches_resolution
#
# swinTransformerBlock = SwinTransformerBlock(dim=int(channel * patchsize ** 2),
#                                             input_resolution=(patches_resolution[0],
#                                                               patches_resolution[1]),
#                                             num_heads=4, window_size=7,
#                                             shift_size=7 // 2,
#                                             mlp_ratio=4.,
#                                             qkv_bias=True,
#                                             drop=0, attn_drop=0,
#                                             norm_layer=nn.LayerNorm,
#                                             ).cuda()
# embed = swinTransformerBlock(embed)
# print(embed.shape)
#
# unpack, _ = unpatch2(unflatten(embed, channel), channel)
# print(unpack.shape)
#
# print(x.shape)

# torch.cuda.reset_peak_memory_stats()
# start_mem = torch.cuda.memory_allocated()
# down = downsample(channel, channel * 2, 4, apply_batchnorm=True, use_residual=True, need_attention=True,
#                   attn_size=imgsize, fp16=True, head=1).cuda()
# summary(down, [x.shape],
#         dtypes=[torch.float])
#
# x, x_no_down = down(x)
# loss = torch.sum(x)
# loss.backward(retain_graph=True)
# end_mem = torch.cuda.memory_allocated()
# peak_mem = torch.cuda.max_memory_allocated()
# print(f"additional memory for training: {(end_mem - start_mem) / 1024 ** 2:.2f} MB")
# print(f"peak memory for training: {peak_mem / 1024 ** 2:.2f} MB")
# parameter_size = 0
# for parameter in down.parameters():
#     parameter_size += parameter.numel()
# print(f"parameter_size:: {parameter_size / 1024 ** 2:.2f} MB")
# print("x.shape", x.shape)
# print("x_no_down.shape", x_no_down.shape)
# print(torch.isnan(x_no_down).any())
# x = torch.zeros([1, 1152 // 2, 3, 3]).cuda()
# down = Downsample_normal(1152 // 2, 1728 // 2, 4, need_attention=3, attn_size=3,
#                          fp16=True, head=9).cuda()

# x, x_no_down = down(x)
# print(x.shape)
# print(x_no_down.shape)
with torch.no_grad():
    encoder = Encoder(scale=2).cuda()

    encoder_optimizer = AdaBelief(encoder.parameters(), lr=2e-6, betas=(0.5, 0.999))
    head_part_optimizer, head_part_scheduler, hand_part_optimizer, hand_part_scheduler, global_part_optimizer, global_part_scheduler = encoder.init_part_optimizer()

    x = torch.randn([1, 3, 448, 448]).cuda()
    x_s = torch.randn([1, 3, 448, 448]).cuda()
    inputs = [x, x_s]
    x, skips = encoder(inputs)

    keypoints_info = torch.randn([1, 512]).cuda()

    inputs = [skips, keypoints_info]
    decoder = Decoder(scale=2).cuda()

    x = decoder(inputs)
    print(x.shape)
