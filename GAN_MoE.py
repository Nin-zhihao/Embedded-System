import math
import time
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

OUTPUT_CHANNELS = 3
device = "cuda"

class PatchEmbed(nn.Module):
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
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
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
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(attn.device))).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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
        return f'dim={self.dim}, window_size={self.window_size}, pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
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
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size
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
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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
            mask_windows = window_partition(img_mask, self.window_size)
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
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
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
        if self.n > 1:
            self.need_pool = True
            head = self.n ** 2
            self.head = head
            self.use_MultiHeadSelfAttention = False
        else:
            self.need_pool = False
            head = min_head
            self.head = head
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.norm_2 = nn.BatchNorm2d(num_features=in_channels)
        if self.need_pool:
            self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.gap = self.n // 4
            self.patch_size = patched_size
            self.split_head_num = self.size ** 2 // self.patch_size ** 2
            self.target_global = patched_size * 4
            self.avg_p = nn.AdaptiveAvgPool2d(output_size=(self.target_global, self.target_global))
            self.up_sample = nn.Upsample(size=(self.size, self.size))
            self.patch_weight = 0.75
            self.global_weight = 1 - self.patch_weight
            self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                self.small_attention = mha.MHA(embed_dim=size ** 2, num_heads=min_head, device='cuda',
                                               use_flash_attn=False, dwconv=True)
            else:
                self.q_group = [nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                          dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda() for _ in range(head)]
                self.k_group = [nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                          dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda() for _ in range(head)]
                self.v_group = [nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                          dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda() for _ in range(head)]
                self.small_proj_out = nn.Conv3d(head, 1, kernel_size=1, stride=1, padding="same", bias=False)

    def cal_bmm_patch(self, q, k, v, b, c):
        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)
        q = q.view(b, c * self.split_head_num, self.patch_size ** 2).permute(0, 2, 1)
        k = k.view(b, c * self.split_head_num, self.patch_size ** 2)
        w_in = torch.bmm(q, k)
        w_in = w_in * (int(c * self.split_head_num) ** (-0.5))
        w_in = F.softmax(w_in, dim=2, dtype=torch.float32)
        v = v.view(b, c * self.split_head_num, self.patch_size ** 2)
        w_in = w_in.permute(0, 2, 1)
        h_in = torch.bmm(v, w_in)
        h_in = h_in.view((b, c, self.size, self.size))
        return h_in

    def cal_bmm_global(self, q, k, v, b, c):
        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)
        q = q.view(b, c, self.target_global ** 2).permute(0, 2, 1)
        k = k.view(b, c, self.target_global ** 2)
        w_in = torch.bmm(q, k)
        w_in = w_in * (int(c) ** (-0.5))
        w_in = F.softmax(w_in, dim=2, dtype=torch.float32)
        v = v.view(b, c, self.target_global ** 2)
        w_in = w_in.permute(0, 2, 1)
        h_in = torch.bmm(v, w_in)
        h_in = h_in.view((b, c, self.target_global, self.target_global))
        h_in = self.up_sample(h_in)
        return h_in

    def cal_bmm(self, q, k, v, b, c, h, w, d_1, h_1, w_1):
        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        q = q.reshape(b * d_1, c, h_1 * w_1).permute(0, 2, 1)
        k = k.reshape(b * d_1, c, h_1 * w_1)
        w_in = torch.bmm(q, k)
        w_in = w_in * (int(c) ** (-0.5))
        w_in = F.softmax(w_in, dim=2, dtype=torch.float32)
        v = v.reshape(b * d_1, c, h_1 * w_1)
        w_in = w_in.permute(0, 2, 1)
        h_in = torch.bmm(v, w_in)
        h_in = h_in.reshape(b, d_1, c, h_1, w_1)
        return h_in

    def call_funtion(self, x):
        h_ = x
        b, c, h, w = h_.shape
        head = self.head
        d_1 = head
        if self.need_pool:
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)
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
        q_list = []
        k_list = []
        v_list = []
        if self.need_pool:
            pass
        elif not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                pass
            else:
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
        if self.mixed_precision:
            if self.need_pool:
                with autocast(dtype=torch.float):
                    h_in_patch = self.cal_bmm_patch(q, k, v, b, c)
                    h_in_global = self.cal_bmm_global(q_p, k_p, v_p, b, c)
                    h_in = self.patch_weight * h_in_patch + self.global_weight * h_in_global
            else:
                if not self.use_MultiHeadSelfAttention:
                    with autocast(dtype=torch.float):
                        h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                else:
                    h_in = self.small_attention(h_)
        else:
            if self.need_pool:
                h_in_patch = self.cal_bmm_patch(q, k, v, b, c)
                h_in_global = self.cal_bmm_global(q_p, k_p, v_p, b, c)
                h_in = self.patch_weight * h_in_patch + self.global_weight * h_in_global
            else:
                if not self.use_MultiHeadSelfAttention:
                    h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                else:
                    h_in = self.small_attention(h_)
        if not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                h_ = h_in.reshape(b, c, h, w)
            else:
                h_ = self.small_proj_out(h_in)
                h_ = h_.reshape(b, c, h, w)
            return h_
        else:
            with autocast(dtype=torch.float):
                h_ = self.proj_out(h_in)
                return h_

    def forward(self, x):
        x_n = self.norm(x)
        if self.mixed_precision:
            with autocast(dtype=torch.bfloat16):
                h_ = self.call_funtion(x_n)
        else:
            h_ = self.call_funtion(x_n)
        return x + h_

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
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, need_attention=False, attn_size=128,
                 fp16=False, activate_fun="LeakyReLU", patched_size=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=fp16, patched_size=patched_size))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True) if activate_fun == "LeakyReLU" else nn.ReLU(inplace=True))
        self.Sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.Sequential(x)

class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, use_residual=True, need_attention=False,
                 attn_size=128, fp16=False, head=4, patch_size=4, embed_dim=96, patch_norm=True, window_size=7,
                 mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        self.patch_norm = patch_norm
        norm_layer = nn.LayerNorm
        self.splitted_in_channel = in_channels // 3
        self.splitted_out_channel = out_channels // 3
        self.patch_embed = PatchEmbed(img_size=attn_size, patch_size=patch_size,
                                      in_chans=self.splitted_in_channel,
                                      embed_dim=self.splitted_in_channel * patch_size ** 2,
                                      norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.SwinTransformerBlock_head = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                              input_resolution=(patches_resolution[0], patches_resolution[1]),
                                                              num_heads=head, window_size=window_size,
                                                              shift_size=window_size // 2,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              drop=0, attn_drop=0,
                                                              norm_layer=norm_layer)
        self.SwinTransformerBlock_hand = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                              input_resolution=(patches_resolution[0], patches_resolution[1]),
                                                              num_heads=head, window_size=window_size,
                                                              shift_size=window_size // 2,
                                                              mlp_ratio=mlp_ratio,
                                                              qkv_bias=qkv_bias,
                                                              drop=0, attn_drop=0,
                                                              norm_layer=norm_layer)
        self.SwinTransformerBlock_global = SwinTransformerBlock(dim=int(self.splitted_in_channel * patch_size ** 2),
                                                                input_resolution=(patches_resolution[0], patches_resolution[1]),
                                                                num_heads=head, window_size=window_size,
                                                                shift_size=window_size // 2,
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias,
                                                                drop=0, attn_drop=0,
                                                                norm_layer=norm_layer)
        self.conv_linear_head = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same', bias=False)
        self.conv_linear_hand = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same', bias=False)
        self.conv_linear_global = nn.Conv2d(in_channels, self.splitted_in_channel, kernel_size=size, padding='same', bias=False)
        self.after_vit_conv2d_hand = nn.Sequential(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=need_attention, attn_size=attn_size, fp16=fp16),
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=False, attn_size=attn_size, fp16=fp16))
        self.residual_hand = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel))
        self.after_vit_conv2d_head = nn.Sequential(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=need_attention, attn_size=attn_size, fp16=fp16),
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=False, attn_size=attn_size, fp16=fp16))
        self.residual_head = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel))
        self.after_vit_conv2d_global = nn.Sequential(
            Attn_Conv2d(in_channels=self.splitted_in_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=need_attention, attn_size=attn_size, fp16=fp16),
            Attn_Conv2d(in_channels=self.splitted_out_channel, out_channels=self.splitted_out_channel, size=size,
                        apply_batchnorm=apply_batchnorm, need_attention=False, attn_size=attn_size, fp16=fp16))
        self.residual_global = nn.Sequential(
            nn.Conv2d(self.splitted_in_channel, out_channels=self.splitted_out_channel, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.splitted_out_channel))
        self.output_conv2d_layer = Attn_Conv2d(in_channels=out_channels, out_channels=out_channels, size=size,
                                               apply_batchnorm=apply_batchnorm, need_attention=False,
                                               attn_size=attn_size, fp16=fp16)
        self.down = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        x_head = self.conv_linear_head(x)
        x_hand = self.conv_linear_hand(x)
        x_global = self.conv_linear_global(x)
        x_head = self.patch_embed(x_head)
        x_hand = self.patch_embed(x_hand)
        x_global = self.patch_embed(x_global)
        x_head = self.SwinTransformerBlock_head(x_head)
        x_hand = self.SwinTransformerBlock_hand(x_hand)
        x_global = self.SwinTransformerBlock_global(x_global)
        x_head, _ = unpatch2(unflatten(x_head, self.splitted_in_channel), self.splitted_in_channel)
        x_hand, _ = unpatch2(unflatten(x_hand, self.splitted_in_channel), self.splitted_in_channel)
        x_global, _ = unpatch2(unflatten(x_global, self.splitted_in_channel), self.splitted_in_channel)
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

class upsample(nn.Module):
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
        self.conv_kernel = 3
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 0:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 1:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 2:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.result = nn.Sequential(*layers)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels))
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True))
        self.weight = torch.nn.Parameter(torch.randn([self.in_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.affine = nn.Sequential(FullyConnectedLayer(z_dim, in_channels, bias_init=1))

    def forward(self, inputs):
        x, keypoints_info = inputs[0], inputs[1]
        if self.use_mapping:
            styles = self.affine(keypoints_info)
            dtype = torch.float16 if (self.use_fp16 and x.device.type == 'cuda') else torch.float32
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
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, activation='linear', bias=True, lr_multiplier=1, weight_init=1, bias_init=0):
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
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class Generator(nn.Module):
    def __init__(self, lr_multiplier=0.01, z_dim=512):
        scale = 2
        super(Generator, self).__init__()
        self.down_1 = nn.Sequential(downsample(3, 64 // scale, 4, apply_batchnorm=False, use_residual=False, need_attention=1,
                                                attn_size=512, fp16=True, head=1))
        self.down_2 = nn.Sequential(downsample(64 // scale, 128 // scale, 4, need_attention=2, attn_size=256, fp16=True, head=3))
        self.down_3 = nn.Sequential(downsample(128 // scale, 256 // scale, 4, need_attention=2, attn_size=128, fp16=True, head=2))
        self.down_4 = nn.Sequential(downsample(256 // scale, 512 // scale, 4, need_attention=2, attn_size=64, fp16=True, head=4))
        self.down_5 = nn.Sequential(downsample(512 // scale, 768 // scale, 4, need_attention=3, attn_size=32, fp16=True, head=4))
        self.down_6 = nn.Sequential(downsample(768 // scale, 1024 // scale, 4, need_attention=3, attn_size=16, fp16=True, head=4))
        self.down_7 = nn.Sequential(downsample(1024 // scale, 1536 // scale, 4, need_attention=3, attn_size=8, fp16=True, head=4))
        self.down_8 = nn.Sequential(downsample(1536 // scale, 2048 // scale, 4, need_attention=3, attn_size=4, fp16=True, head=4))
        self.down_9 = nn.Sequential(downsample(2048 // scale, 4096 // scale, 2, apply_batchnorm=True, use_residual=True, need_attention=3, attn_size=2, fp16=True, head=4))
        self.down_stack = [self.down_1, self.down_2, self.down_3, self.down_4,
                           self.down_5, self.down_6, self.down_7, self.down_8, self.down_9]
        self.up_1 = nn.Sequential(upsample(4096 // scale, 2048 // scale, 2, apply_dropout=True,
                                            need_attention=3, attn_size=2, atten_fp16=True, head=4))
        self.up_2 = nn.Sequential(upsample(4096 // scale, 1536 // scale, 4, apply_dropout=True,
                                            need_attention=3, attn_size=4, atten_fp16=True, head=4))
        self.up_3 = nn.Sequential(upsample(3072 // scale, 1024 // scale, 4, apply_dropout=True,
                                            need_attention=3, attn_size=8, atten_fp16=True, head=4))
        self.up_4 = nn.Sequential(upsample(2048 // scale, 768 // scale, 4, need_attention=3, attn_size=16, atten_fp16=True, head=4))
        self.up_5 = nn.Sequential(upsample(1536 // scale, 512 // scale, 4, need_attention=3, attn_size=32, atten_fp16=True, head=4))
        self.up_6 = nn.Sequential(upsample(1024 // scale, 256 // scale, 4, need_attention=2, attn_size=64, atten_fp16=True, head=4))
        self.up_7 = nn.Sequential(upsample(512 // scale, 128 // scale, 4, need_attention=2, attn_size=128, atten_fp16=True, head=2))
        self.up_8 = nn.Sequential(upsample(256 // scale, 64 // scale, 4, need_attention=2, attn_size=256, atten_fp16=True, head=2))
        self.up_stack = [self.up_1, self.up_2, self.up_3, self.up_4,
                         self.up_5, self.up_6, self.up_7, self.up_8]
        self.mapping_network = nn.Sequential(FullyConnectedLayer(266, z_dim),
                                              FullyConnectedLayer(z_dim, z_dim, activation='lrelu', lr_multiplier=lr_multiplier),
                                              FullyConnectedLayer(z_dim, z_dim, activation='lrelu', lr_multiplier=lr_multiplier))
        self.final_up = nn.Sequential(upsample(128 // scale, 32 // scale, 4, need_attention=1, attn_size=512, atten_fp16=True, head=1))
        self.out_layer = nn.Sequential(nn.Conv2d(32 // scale, OUTPUT_CHANNELS, kernel_size=1, padding="same", bias=False),
                                       nn.Tanh())
        self.ada_in = AdaIN()
        self.down_Adain = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, inputs_content, inputs_style, keypoint_input_bodies, keypoint_input_faces,
                keypoint_input_left_hands, keypoint_input_right_hands):
        x = inputs_content
        x_s = inputs_style
        bodies = nn.Flatten()(keypoint_input_bodies)
        faces = nn.Flatten()(keypoint_input_faces)
        left_hands = nn.Flatten()(keypoint_input_left_hands)
        right_hands = nn.Flatten()(keypoint_input_right_hands)
        keypoints_info = torch.concat([bodies, faces, left_hands, right_hands], dim=1)
        keypoints_info_f = self.mapping_network(keypoints_info)
        skips = []
        for index in range(len(self.down_stack)):
            x, x_no_down = self.down_stack[index](x)
            x_s, x_no_down_s = self.down_stack[index](x_s)
            x_skip = self.ada_in(x_no_down, x_no_down_s, False)
            x_skip = self.down_Adain(x_skip)
            skips.append(x_skip)
        x = skips[-1]
        for index in range(len(self.up_stack)):
            inputs_up = (x, keypoints_info_f)
            x = self.up_stack[index](inputs_up)
            skip_in = skips[len(self.up_stack) - 1 - index]
            x = torch.concat([x, skip_in], dim=1)
        inputs_final = (x, keypoints_info_f)
        x = self.final_up(inputs_final)
        x = self.out_layer(x)
        return x

class downsample_d(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, need_down=True, need_attention=True,
                 attn_size=128, atten_fp16=False):
        super().__init__()
        self.need_attention = need_attention
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=1, padding="same", bias=False))
        if need_attention:
            layers.append(AttnBlock6_b(out_channels, attn_size, min_head=2, fp16=atten_fp16, force_bmm_batch=False))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))
        if need_down:
            layers.append(nn.MaxPool2d(stride=2, kernel_size=2))
        self.result = nn.Sequential(*layers)

    def forward(self, x):
        return self.result(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 512
        self.downsample_d_1 = downsample_d(6, 128, 4, need_attention=False, attn_size=512, atten_fp16=True)
        self.downsample_d_2 = downsample_d(128, 256, 4, need_attention=True, attn_size=256, atten_fp16=True)
        self.downsample_d_3 = downsample_d(256, 384, 4, need_attention=True, attn_size=128, atten_fp16=True)
        self.downsample_d_4 = downsample_d(384, out_channels, 4, need_attention=True, attn_size=64, atten_fp16=True)
        self.conv_out1 = nn.Sequential(nn.ZeroPad2d(1),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
                                       nn.GroupNorm(num_groups=1, num_channels=out_channels),
                                       nn.LeakyReLU(inplace=True),
                                       nn.ZeroPad2d(1),
                                       nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, inp, tar):
        x = torch.concat([inp, tar], dim=1)
        x = self.downsample_d_1(x)
        x = self.downsample_d_2(x)
        x = self.downsample_d_3(x)
        x = self.downsample_d_4(x)
        x = self.conv_out1(x)
        return x

class Discriminator_Hand(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 512
        self.downsample_d_1 = downsample_d(6, 128, 4, need_attention=False, attn_size=128, atten_fp16=True)
        self.downsample_d_2 = downsample_d(128, 256, 4, need_down=False, need_attention=True, attn_size=64, atten_fp16=True)
        self.downsample_d_3 = downsample_d(256, 384, 4, need_down=False, need_attention=True, attn_size=64, atten_fp16=True)
        self.downsample_d_4 = downsample_d(384, out_channels, 4, need_attention=True, attn_size=64, atten_fp16=True)
        self.conv_out2 = nn.Sequential(nn.ZeroPad2d(1),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
                                       nn.GroupNorm(num_groups=1, num_channels=out_channels),
                                       nn.LeakyReLU(inplace=True),
                                       nn.ZeroPad2d(1),
                                       nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, inp, tar):
        x = torch.concat([inp, tar], dim=1)
        x = self.downsample_d_1(x)
        x = self.downsample_d_2(x)
        x = self.downsample_d_3(x)
        x = self.downsample_d_4(x)
        x = self.conv_out2(x)
        return x

class Discriminator_Head(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 512
        self.downsample_d_1 = downsample_d(6, 128, 4, need_attention=False, attn_size=128, atten_fp16=True)
        self.downsample_d_2 = downsample_d(128, 256, 4, need_down=False, need_attention=True, attn_size=64, atten_fp16=True)
        self.downsample_d_3 = downsample_d(256, 384, 4, need_down=False, need_attention=True, attn_size=64, atten_fp16=True)
        self.downsample_d_4 = downsample_d(384, out_channels, 4, need_attention=True, attn_size=64, atten_fp16=True)
        self.conv_out2 = nn.Sequential(nn.ZeroPad2d(1),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
                                       nn.GroupNorm(num_groups=1, num_channels=out_channels),
                                       nn.LeakyReLU(inplace=True),
                                       nn.ZeroPad2d(1),
                                       nn.Conv2d(out_channels, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, inp, tar):
        x = torch.concat([inp, tar], dim=1)
        x = self.downsample_d_1(x)
        x = self.downsample_d_2(x)
        x = self.downsample_d_3(x)
        x = self.downsample_d_4(x)
        x = self.conv_out2(x)
        return x

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

torch.cuda.reset_peak_memory_stats()
start_mem = torch.cuda.memory_allocated()
channel = 27
imgsize = 448
patchsize = 4
x = torch.zeros([1, channel, imgsize, imgsize]).cuda()
down = downsample(channel, channel * 2, 4, apply_batchnorm=True, use_residual=True, need_attention=True,
                  attn_size=imgsize, fp16=True, head=1).cuda()
from torchinfo import summary
summary(down, [x.shape], dtypes=[torch.float])
x, x_no_down = down(x)
loss = torch.sum(x)
loss.backward(retain_graph=True)
end_mem = torch.cuda.memory_allocated()
peak_mem = torch.cuda.max_memory_allocated()
print(f"additional memory for training: {(end_mem - start_mem) / 1024**2:.2f} MB")
print(f"peak memory for training: {peak_mem / 1024**2:.2f} MB")
parameter_size = sum(p.numel() for p in down.parameters())
print(f"parameter_size:: {parameter_size / 1024**2:.2f} MB")
print("x.shape", x.shape)
print("x_no_down.shape", x_no_down.shape)
print(torch.isnan(x_no_down).any())