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

OUTPUT_CHANNELS = 3

from FlashAttention2 import attention
from FlashAttention2.attention import FlashAttention
from self_attention_cv import MultiHeadSelfAttention


class AttnBlock(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
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
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        n_channels = in_channels // 8
        self.n_channels = n_channels
        self.small_q = torch.nn.Conv3d(n_channels,
                                       n_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.small_k = torch.nn.Conv3d(n_channels,
                                       n_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_v = torch.nn.Conv3d(n_channels,
                                       n_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_proj_out = torch.nn.Conv3d(n_channels,
                                              n_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)

        self.n = int(size // 64)
        if self.n > 0:
            self.need_pool = True
        else:
            self.need_pool = False
        if self.need_pool:
            self.avgp_3d = torch.nn.AvgPool3d((1, self.n, self.n), (1, self.n, self.n))
            self.upsample = torch.nn.Upsample(scale_factor=(1, self.n, self.n), mode='nearest')

    def forward(self, x):
        # print("attn")
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        d_1 = 8
        c_1 = c // 8
        if self.n > 0:
            h_1 = h // self.n
            w_1 = w // self.n
        else:
            h_1 = h
            w_1 = w

        q = q.reshape(b, c_1, d_1, h, w)
        k = k.reshape(b, c_1, d_1, h, w)
        v = v.reshape(b, c_1, d_1, h, w)

        q = self.small_q(q)
        k = self.small_k(k)
        v = self.small_v(v)
        if self.need_pool:
            q = self.avgp_3d(q)
            k = self.avgp_3d(k)
            v = self.avgp_3d(v)

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        # print(q.shape)
        h_in_groups = []
        for i in range(8):
            q_in = q[:, i, :, :, :]
            k_in = k[:, i, :, :, :]
            v_in = v[:, i, :, :, :]

            q_in = q_in.reshape(b, c_1, h_1 * w_1)
            q_in = q_in.permute(0, 2, 1)  # b,hw,c
            k_in = k_in.reshape(b, c_1, h_1 * w_1)  # b,c,hw
            w_in = torch.bmm(q_in, k_in)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_in = w_in * (int(c_1) ** (-0.5))
            w_in = torch.nn.functional.softmax(w_in, dim=2)

            # attend to values
            v_in = v_in.reshape(b, c_1, h_1 * w_1)
            w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            h_in = torch.bmm(v_in, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_in = h_in.reshape(b, c_1, h_1, w_1)
            # print(h_in.shape)
            h_in_groups.append(h_in)

        h_ = torch.stack(h_in_groups, dim=1)
        h_ = h_.permute(0, 2, 1, 3, 4)
        # print(h_.shape)
        if self.need_pool:
            h_ = self.upsample(h_)
        # print(h_.shape)
        h_ = self.small_proj_out(h_)
        # print(h_.shape)
        h_ = h_.reshape(b, c, h, w)
        # print(h_.shape)
        h_ = self.proj_out(h_)
        return x + h_


class AttnBlock2(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.in_channels = in_channels
        head = 4
        self.head = head
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
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
        self.proj_out = torch.nn.Conv2d(in_channels * head,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        n_channels = in_channels // head
        self.n_channels = n_channels
        self.small_q = torch.nn.Conv3d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.small_k = torch.nn.Conv3d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_v = torch.nn.Conv3d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_proj_out = torch.nn.Conv3d(in_channels,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)

        self.enhance_layer_group = [torch.nn.Conv3d(in_channels,
                                                    in_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0), ]

        self.n = int(size // 32)
        if self.n > 0:
            self.need_pool = True
        else:
            self.need_pool = False
        if self.need_pool:
            self.avgp_3d = torch.nn.AvgPool3d((1, self.n, self.n), (1, self.n, self.n))
            self.upsample = torch.nn.Upsample(scale_factor=(1, self.n, self.n), mode='trilinear')

    def forward(self, x):
        # print("attn")
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        head = self.head
        d_1 = head
        c_1 = c // head
        if self.n > 0:
            h_1 = h // self.n
            w_1 = w // self.n
        else:
            h_1 = h
            w_1 = w
        q = torch.stack([q for i in range(head)], dim=2)
        k = torch.stack([k for i in range(head)], dim=2)
        v = torch.stack([v for i in range(head)], dim=2)

        # q = q.reshape(b, c_1, d_1, h, w)
        # k = k.reshape(b, c_1, d_1, h, w)
        # v = v.reshape(b, c_1, d_1, h, w)
        if self.need_pool:
            q = self.avgp_3d(q)
            k = self.avgp_3d(k)
            v = self.avgp_3d(v)

        q = self.small_q(q)
        k = self.small_k(k)
        v = self.small_v(v)

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        # print(q.shape)
        h_in_groups = []
        for i in range(head):
            q_in = q[:, i, :, :, :]
            k_in = k[:, i, :, :, :]
            v_in = v[:, i, :, :, :]

            q_in = q_in.reshape(b, c, h_1 * w_1)
            q_in = q_in.permute(0, 2, 1)  # b,hw,c
            k_in = k_in.reshape(b, c, h_1 * w_1)  # b,c,hw
            w_in = torch.bmm(q_in, k_in)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_in = w_in * (int(c) ** (-0.5))
            w_in = torch.nn.functional.softmax(w_in, dim=2)

            # attend to values
            v_in = v_in.reshape(b, c, h_1 * w_1)
            w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            h_in = torch.bmm(v_in, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_in = h_in.reshape(b, c, h_1, w_1)
            # print(h_in.shape)
            h_in_groups.append(h_in)

        h_ = torch.stack(h_in_groups, dim=1)
        h_ = h_.permute(0, 2, 1, 3, 4)
        # print(h_.shape)
        if self.need_pool:
            h_ = self.upsample(h_)
        # print(h_.shape)
        h_ = self.small_proj_out(h_)
        # print(h_.shape)
        h_ = h_.reshape(b, c * d_1, h, w)
        # print(h_.shape)
        h_ = self.proj_out(h_)
        return x + h_


class AttnBlock3(nn.Module):
    def __init__(self, in_channels, size, min_head=4, fp16=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.n = int(size // 32)
        # self.max_size=256
        # self.max_n=size//self.max_size

        if self.n > 0:
            self.need_pool = True
            head = self.n
            self.head = head
            if self.n < 4:
                head = min_head
                self.head = head
        else:
            self.need_pool = False
            head = min_head
            self.head = head
        self.num_worker = 4
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        # self.norm_out = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # self.q = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        # self.k = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        # self.v = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # n_channels = in_channels // head
        # self.n_channels = n_channels
        self.small_q = torch.nn.Conv3d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.small_k = torch.nn.Conv3d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_v = torch.nn.Conv3d(in_channels,
                                       in_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_proj_out = torch.nn.Conv3d(head,
                                              1,
                                              kernel_size=1,
                                              stride=1,
                                              padding="same")
        # if self.max_n>1:
        #     self.avgp=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        if self.need_pool:
            self.upsample = torch.nn.Upsample(scale_factor=(self.n, self.n), mode='bicubic')

    def job(self, q_in, k_in, v_in, b, c, h_1, w_1):

        q_in = q_in.reshape(b, c, h_1 * w_1)
        q_in = q_in.permute(0, 2, 1)  # b,hw,c
        k_in = k_in.reshape(b, c, h_1 * w_1)  # b,c,hw
        w_in = torch.bmm(q_in, k_in)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_in = w_in * (int(c) ** (-0.5))
        w_in = torch.nn.functional.softmax(w_in, dim=2)

        # attend to values
        v_in = v_in.reshape(b, c, h_1 * w_1)
        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_in = torch.bmm(v_in, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_in = h_in.reshape(b, c, h_1, w_1)
        # print(h_in.shape)
        if self.need_pool:
            h_in = self.upsample(h_in)
        return h_in

    def cal_bmm(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

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

        h_in = h_in.reshape(b * d_1, c, h_1, w_1)
        # print("h_in:",h_in.dtype)
        return h_in

    def call_funtion(self, x):
        h_ = x
        q = h_
        k = h_
        v = h_

        # compute attention
        b, c, h, w = q.shape
        head = self.head
        d_1 = head
        c_1 = c // head
        if self.n > 0:
            h_1 = h // self.n
            w_1 = w // self.n
        else:
            h_1 = h
            w_1 = w
        # q=torch.stack([q for i in range(head)],dim=2)
        # k=torch.stack([k for i in range(head)],dim=2)
        # v=torch.stack([v for i in range(head)],dim=2)

        # q = q.reshape(b, c_1, d_1, h, w)
        # k = k.reshape(b, c_1, d_1, h, w)
        # v = v.reshape(b, c_1, d_1, h, w)
        # if self.need_pool:
        #     q = self.avgp_3d(q)
        #     k = self.avgp_3d(k)
        #     v = self.avgp_3d(v)

        q_list = []
        k_list = []
        v_list = []
        if self.need_pool:
            for i in range(self.n):
                q_s = q[:, :, i::self.n, i::self.n]
                k_s = q[:, :, i::self.n, i::self.n]
                v_s = q[:, :, i::self.n, i::self.n]
                q_list.append(q_s)
                k_list.append(k_s)
                v_list.append(v_s)

                # print(q_s.shape)
            q = torch.stack(q_list, dim=2)
            k = torch.stack(k_list, dim=2)
            v = torch.stack(v_list, dim=2)
            if self.n < head:
                q = torch.concat([q for i in range(head // self.n)], dim=2)
                k = torch.concat([k for i in range(head // self.n)], dim=2)
                v = torch.concat([v for i in range(head // self.n)], dim=2)
        else:
            q = torch.stack([q for i in range(head)], dim=2)
            k = torch.stack([k for i in range(head)], dim=2)
            v = torch.stack([v for i in range(head)], dim=2)

        q = self.small_q(q)
        k = self.small_k(k)
        v = self.small_v(v)
        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.float32):
                h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
            # h_in = h_in.to(torch.float32)
        else:
            h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
        # print(head)

        # q = q.permute(0, 2, 1, 3, 4)
        # k = k.permute(0, 2, 1, 3, 4)
        # v = v.permute(0, 2, 1, 3, 4)
        #
        # q = q.reshape(b * d_1, c, h_1 * w_1)
        # q = q.permute(0, 2, 1)  # b,hw,c
        # k = k.reshape(b * d_1, c, h_1 * w_1)  # b,c,hw
        # # q.to(torch.float8_e4m3fn)
        # # k.to(torch.float8_e4m3fn)
        # w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # # w_in.to(torch.float32)
        # w_in = w_in * (int(c) ** (-0.5))
        #
        # w_in = torch.nn.functional.softmax(w_in, dim=2)
        # # w_in.to(torch.float8_e4m3fn)
        #
        # # attend to values
        # v = v.reshape(b * d_1, c, h_1 * w_1)
        # w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # # v.to(torch.float8_e4m3fn)
        #
        # h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        #
        # h_in = h_in.reshape(b * d_1, c, h_1, w_1)
        # h_in.to(torch.float32)

        if self.need_pool:
            h_in = self.upsample(h_in)

        # print(q.shape)
        h_in_groups_result_list = []
        # pool = Pool(self.num_worker)
        # for i in range(head):
        #     q_in = q[:, i, :, :, :]
        #     k_in = k[:, i, :, :, :]
        #     v_in = v[:, i, :, :, :]

        # h_in_groups_result_list.append(pool.apply_async(func=self.job, args=(q_in, k_in, v_in, b, c, h_1, w_1)))

        # q_in = q_in.reshape(b, c, h_1 * w_1)
        # q_in = q_in.permute(0, 2, 1)  # b,hw,c
        # k_in = k_in.reshape(b, c, h_1 * w_1)  # b,c,hw
        # w_in = torch.bmm(q_in, k_in)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_in = w_in * (int(c) ** (-0.5))
        # w_in = torch.nn.functional.softmax(w_in, dim=2)
        #
        # # attend to values
        # v_in = v_in.reshape(b, c, h_1 * w_1)
        # w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # h_in = torch.bmm(v_in, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_in = h_in.reshape(b, c, h_1, w_1)
        # # print(h_in.shape)
        # if self.need_pool:
        #     h_in = self.upsample(h_in)
        # h_in_groups.append(h_in)
        # pool.close()
        # pool.join()
        # h_in_groups.append(h_in)

        # h_in_groups = []
        # for i in h_in_groups_result_list:
        #     h_in_groups.append(i.get())
        # h_in = self.norm(h_in)
        h_in = h_in.reshape(b, d_1, c, h, w)
        # if self.mixed_precision:
        #     # Use autocast to allow operations to run in lower precision
        #     with autocast(dtype=torch.bfloat16):
        #         h_ = self.small_proj_out(h_in)
        #         h_ = h_.reshape(b, c, h, w)
        #         h_ = h_.to(torch.float32)
        # else:
        h_ = self.small_proj_out(h_in)
        h_ = h_.reshape(b, c, h, w)

        # h_ = self.norm(h_)

        # h_ = torch.stack(h_in_groups, dim=1)
        # h_ = h_.permute(0, 2, 1, 3, 4)
        # print(h_.shape)
        # if self.need_pool:
        #     h_ = self.upsample(h_)
        # print(h_.shape)
        # print(h_.shape)
        # print(h_.shape)
        # h_ = self.proj_out(h_)
        # h_ = self.norm(h_)
        return h_

    def forward(self, x):
        h = self.norm(x)
        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.bfloat16):
                h_ = self.call_funtion(h)
        else:
            h_ = self.call_funtion(h)
        # print("attn")

        return x + h_


class AttnBlock4(nn.Module):
    def __init__(self, in_channels, size, min_head=4, fp16=True):
        super().__init__()
        self.in_channels = in_channels
        self.n = int(size // 128)
        if self.n > 0:
            self.need_pool = True
            head = self.n
            self.head = head
            if self.n < 4:
                head = min_head
                self.head = head
        else:
            self.need_pool = False
            head = min_head
            self.head = head
        self.num_worker = 4
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.flash_attention = FlashAttention(dim=in_channels, heads=4, parallel=False, mixed_precision=fp16)

        if self.need_pool:
            self.maxp = torch.nn.MaxPool2d(kernel_size=self.n, stride=self.n)
            self.upsample = torch.nn.Upsample(scale_factor=(self.n, self.n), mode='bicubic')
            self.proj_out = torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

    def forward(self, x):
        # print("attn")
        h_ = x
        h_ = self.norm(h_)
        if self.need_pool:
            h_ = self.maxp(h_)
        b, c, h, w = h_.shape
        h_ = h_.permute(0, 2, 3, 1)
        h_ = h_.reshape(b, h * w, c)
        h_ = self.flash_attention(h_)
        print(h_.shape)
        h_ = h_.reshape(b, h, w, c)
        h_ = h_.permute(0, 3, 1, 2)
        if self.need_pool:
            h_ = self.upsample(h_)
            h_ = self.proj_out(h_)

        print(h_.shape)

        return x + h_


class AttnBlock5(nn.Module):
    def __init__(self, in_channels, size, head=4, fp16=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.n_length = int(128 * 128 // head)
        self.n = size ** 2 // (self.n_length * head)
        # self.max_size=256
        # self.max_n=size//self.max_size

        # if self.n > 0:
        #     self.need_pool = True
        #     head = self.n
        #     self.head = head
        #     if self.n < 4:
        #         head = min_head
        #         self.head = head
        # else:
        #     self.need_pool = False
        #     head = min_head
        #     self.head = head
        # self.num_worker = 4
        self.head = head
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        # self.norm_out = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # self.q = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        # self.k = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        # self.v = torch.nn.Conv2d(in_channels,
        #                          in_channels,
        #                          kernel_size=1,
        #                          stride=1,
        #                          padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # n_channels = in_channels // head
        # self.n_channels = n_channels
        self.small_q = torch.nn.Conv3d(1,
                                       head,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        self.small_k = torch.nn.Conv3d(1,
                                       head,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_v = torch.nn.Conv3d(1,
                                       head,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        self.small_proj_out = torch.nn.Conv3d(head,
                                              1,
                                              kernel_size=1,
                                              stride=1,
                                              padding="same")
        # if self.max_n>1:
        #     self.avgp=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        # if self.need_pool:
        #     self.upsample = torch.nn.Upsample(scale_factor=(self.n, self.n), mode='bicubic')

    def job(self, q_in, k_in, v_in, b, c, h_1, w_1):

        q_in = q_in.reshape(b, c, h_1 * w_1)
        q_in = q_in.permute(0, 2, 1)  # b,hw,c
        k_in = k_in.reshape(b, c, h_1 * w_1)  # b,c,hw
        w_in = torch.bmm(q_in, k_in)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_in = w_in * (int(c) ** (-0.5))
        w_in = torch.nn.functional.softmax(w_in, dim=2)

        # attend to values
        v_in = v_in.reshape(b, c, h_1 * w_1)
        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_in = torch.bmm(v_in, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_in = h_in.reshape(b, c, h_1, w_1)
        # print(h_in.shape)
        if self.need_pool:
            h_in = self.upsample(h_in)
        return h_in

    def cal_bmm_batch(self, q, k, v, b, c, h, w, d_1):

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # print("q:",q.dtype)
        print(self.n_length)
        print(self.n)

        q = q.reshape(b * d_1, c, h * w // self.n, self.n)
        k = k.reshape(b * d_1, c, h * w // self.n, self.n)  # b,c,hw

        w_in_list = []
        for i in range(self.n):
            w_in_in_list = []
            for n in range(self.n):
                q_in = q[:, :, :, i]
                k_in = q[:, :, :, i]
                q_in = q_in.permute(0, 2, 1)  # b,hw,c
                # q.to(torch.float8_e4m3fn)
                # k.to(torch.float8_e4m3fn)
                print(q_in.shape)
                print(k_in.shape)
                print("i:", i)
                print("n:", n)

                w_in = torch.bmm(q_in, k_in)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
                w_in_in_list.append(w_in)
            w_in_list_c = torch.concat(w_in_in_list, dim=2)
            # print("w_in:",w_in.dtype)
            w_in_list.append(w_in_list_c)
            # w_in=w_in.to(torch.float32)
        w_ = torch.concat(w_in_list, dim=1)
        w_ = w_ * (int(c) ** (-0.5))

        w_ = torch.nn.functional.softmax(w_, dim=2, dtype=torch.float32)
        # w_in.to(torch.float8_e4m3fn)

        # attend to values
        v = v.reshape(b * d_1, c, h * w)
        w_ = w_.permute(0, 2, 1)
        w_ = w_.reshape(b * d_1, h * w, h * w // self.n, self.n)
        h_in_list = []
        for m in range(c):
            h_in_in_list = []
            for i in range(self.n):
                v_in = w_[:, m, :, :]

                w_in = w_[:, :, :, i]
                h_in = torch.bmm(v_in, w_in)
                h_in_in_list.append(h_in)
            h_in_list_c = torch.concat(h_in_in_list, dim=2)
            h_in_list.append(h_in_list_c)
        h_ = torch.concat(h_in_list, dim=1)
        h_ = h_.reshape(b * d_1, c, h, w)
        print("cal_bmm_batch h_:", h_.dtype)
        return h_

    def cal_bmm(self, q, k, v, b, c, h, w, d_1):

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # print("q:",q.dtype)

        q = q.reshape(b * d_1, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * d_1, c, h * w)  # b,c,hw
        # q.to(torch.float8_e4m3fn)
        # k.to(torch.float8_e4m3fn)
        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print("w_in:",w_in.dtype)
        # w_in=w_in.to(torch.float32)
        w_in = w_in * (int(c) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)
        # w_in.to(torch.float8_e4m3fn)

        # attend to values
        v = v.reshape(b * d_1, c, h * w)
        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # v.to(torch.float8_e4m3fn)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_in = h_in.reshape(b * d_1, c, h, w)
        # print("h_in:",h_in.dtype)
        print("cal_bmm h_:", h_in.dtype)

        return h_in

    def call_funtion(self, x):
        h_ = x
        q = h_
        k = h_
        v = h_

        # compute attention
        b, c, h, w = q.shape
        head = self.head
        d_1 = head

        q = q.reshape(b, 1, c, h, w)
        k = k.reshape(b, 1, c, h, w)
        v = v.reshape(b, 1, c, h, w)

        q = self.small_q(q)
        k = self.small_k(k)
        v = self.small_v(v)
        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.float32):
                if self.n > 0:
                    h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1)
                else:
                    h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1)

            # h_in = h_in.to(torch.float32)
        else:
            if self.n > 0:
                h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1)
            else:
                h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1)

        h_in = h_in.reshape(b, d_1, c, h, w)
        # if self.mixed_precision:
        #     # Use autocast to allow operations to run in lower precision
        #     with autocast(dtype=torch.bfloat16):
        #         h_ = self.small_proj_out(h_in)
        #         h_ = h_.reshape(b, c, h, w)
        #         h_ = h_.to(torch.float32)
        # else:
        h_ = self.small_proj_out(h_in)
        h_ = h_.reshape(b, c, h, w)

        return h_

    def forward(self, x):
        h = self.norm(x)
        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.bfloat16):
                h_ = self.call_funtion(h)
        else:
            h_ = self.call_funtion(h)
        # print("attn")

        return x + h_


class AttnBlock6(nn.Module):
    def __init__(self, in_channels, size, min_head=4, fp16=False, force_bmm_batch=False, test=False,
                 use_MultiHeadSelfAttention=False, svd=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.n = int(size // 32)
        self.big_n = int(size // 128)
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
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0, bias=False)
        if not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                self.small_attention = FlashAttention(dim=size ** 2, heads=min_head, mixed_precision=False)
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

            # n_channels = in_channels // head
            # self.n_channels = n_channels
            # self.small_q = torch.nn.Conv3d(in_channels,
            #                                in_channels,
            #                                kernel_size=1,
            #                                stride=1,
            #                                padding=0)
            #
            # self.small_k = torch.nn.Conv3d(in_channels,
            #                                in_channels,
            #                                kernel_size=1,
            #                                stride=1,
            #                                padding=0)
            # self.small_v = torch.nn.Conv3d(in_channels,
            #                                in_channels,
            #                                kernel_size=1,
            #                                stride=1,
            #                                padding=0)
            self.small_proj_out = torch.nn.Conv3d(head,
                                                  1,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding="same", bias=False)
        if self.big_n > 0 and not self.force_bmm_batch:
            f_a = 1.09794384566096
            f_b = 0.00172673315982637
            compensate = f_a * torch.exp(torch.Tensor([f_b * in_channels]))
            scale_k = (2.718281828459045 / compensate) / (int(in_channels) ** (-0.5))
            self.scale_k = scale_k.cuda()
            self.svd_bmm_softmax = torch.nn.Softmax(dim=0)
            self.svd_method = "gesvdj" if in_channels > 0 else "gesvda"

        # if self.max_n > 1:
        #     self.avgp = torch.nn.AvgPool2d(kernel_size=(1, 33), stride=(1, 32))
        # if self.need_pool:
        #     self.upsample = torch.nn.Upsample(scale_factor=(self.n, self.n), mode='bicubic')

    def cal_bmm_batch(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # print("q:",q.dtype)
        # print("q:",q.shape)

        q = q.reshape(b * d_1, c, h_1 * w_1)
        q = q.permute(0, 2, 1)  # b,hw,c
        # print("w_in:",w_in.dtype)
        # print("size:",self.size,"q:",q.shape)
        # k = self.avgp(k)
        k = k.reshape(b * d_1, c, h_1 * w_1)  # b,c,hw

        # print("k:",k.shape)

        # q.to(torch.float8_e4m3fn)
        # k.to(torch.float8_e4m3fn)
        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print("w_in:",w_in.dtype)
        # w_in=w_in.to(torch.float32)
        w_in = w_in * (int(c) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)
        # w_in.to(torch.float8_e4m3fn)
        # print("w_in:", w_in.shape)
        # attend to values
        v = v.reshape(b * d_1, c, h_1 * w_1)
        # print("v:", v.shape)

        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # v.to(torch.float8_e4m3fn)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # print("h_in:", h_in.shape)

        h_in = h_in.reshape(b, d_1, c, h_1 * w_1)
        h_in_list = h_in.split(dim=1, split_size=1)
        # print(q_list)
        # cc = torch.stack(q_list, dim=4).view(2, 3, 16)
        h_in = torch.stack(h_in_list, dim=4).view(b, c, d_1 * h_1 * w_1)
        h_in = h_in.reshape(b, c, h, w)
        # print("bmm b h_in:",h_in.dtype)
        # print("bmm b h_in:",h_in.shape)

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

    def cal_bmm_svd(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # time_start = time.time()
        #

        #
        # print(q.shape)
        # print(q)

        q = q.reshape(b, c, h * w)
        # k = k.reshape(b, c, h * w)  # b,c,hw
        v = v.reshape(b, c, h * w)  # b,c,hw

        k = k.reshape([b, -1])
        k = (k - 0.5) * 10
        k = 0.5 * (torch.nn.functional.tanh(k) + 1)
        # print(k)
        # k = self.svd_bmm_softmax(k)
        k = torch.nn.functional.softmax(k, dim=1, dtype=torch.float32)

        # print(k)
        k = k.reshape([b, c, h * w]) * self.scale_k * 10e4
        # print(k)
        # print(self.size)
        # print(self.use_MultiHeadSelfAttention)

        u_q, s_q, v_q = torch.linalg.svd(q, full_matrices=False, driver=self.svd_method)
        u_k, s_k, v_k = torch.linalg.svd(k, full_matrices=False, driver=self.svd_method)
        u_v, s_v, v_v = torch.linalg.svd(v, full_matrices=False, driver=self.svd_method)
        # time_end = time.time()

        # print("size:",self.size,'svd time cost', time_end - time_start, 's')
        # time_start = time.time()
        clip_channle = 4

        u_q = u_q[:, :, 0:clip_channle]
        s_q = s_q[:, 0:clip_channle]
        v_q = v_q[:, 0:clip_channle, :]

        u_k = u_k[:, :, 0:clip_channle]
        s_k = s_k[:, 0:clip_channle]
        v_k = v_k[:, 0:clip_channle, :]

        u_v = u_v[:, :, 0:clip_channle]
        s_v = s_v[:, 0:clip_channle]
        v_v = v_v[:, 0:clip_channle, :]

        sv_q = torch.bmm(torch.diag_embed(s_q), torch.transpose(u_q, 1, dim1=2))
        sv_k = torch.bmm(u_k, torch.diag_embed(s_k))

        # print(sv_q.shape)
        # print(sv_k.shape)

        suus_qk = torch.bmm(sv_q, sv_k)

        # print(suus_qk.shape)

        v_q = v_q * ((int(c) ** (-0.5)) / 10e4)

        vv_vq = torch.bmm(v_v, torch.transpose(v_q, 1, dim1=2))

        # print(vv_vq.shape)

        us_v = torch.bmm(u_v, torch.diag_embed(s_v))
        # print(us_v.shape)

        usvv_vq = torch.bmm(us_v, vv_vq)
        # print(usvv_vq.shape)

        usvv_suus = torch.bmm(usvv_vq, suus_qk)
        # print(usvv_suus.shape)
        # usvv_suus = usvv_suus / (4 * self.size ** 2)

        final_result = torch.bmm(usvv_suus, v_k)
        final_result = final_result.reshape(b, c, h, w)
        final_result = self.norm_2(final_result) / 10e2
        # time_end = time.time()
        # print('bmm time cost', time_end - time_start, 's')

        return final_result

    def cal_bmm_no_svd(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # time_start = time.time()
        #

        #
        # print(q.shape)
        # print(q)

        q = q.reshape(b, c, h * w)
        # k = k.reshape(b, c, h * w)  # b,c,hw
        v = v.reshape(b, c, h * w)  # b,c,hw

        k = k.reshape([b, -1])
        k = (k - 0.5) * 10
        k = 0.5 * (torch.nn.functional.tanh(k) + 1)
        # print(k)
        # k = self.svd_bmm_softmax(k)
        k = torch.nn.functional.softmax(k, dim=1, dtype=torch.float32)

        # print(k)
        k = k.reshape([b, c, h * w]) * self.scale_k
        q = q.permute(0, 2, 1)
        q = q * (int(c) ** (-0.5))
        vq_result = torch.bmm(v, q)
        final_result = torch.bmm(vq_result, k)
        final_result = final_result.reshape(b, c, h, w)
        final_result = self.norm_2(final_result) / 10e2
        # print(final_result)
        # time_end = time.time()
        # print('bmm time cost', time_end - time_start, 's')
        return final_result

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
            h_1 = h // self.n
            w_1 = w // self.n
        else:
            # if not self.use_MultiHeadSelfAttention:
            h_1 = h
            w_1 = w
            q = h_
            k = h_
            v = h_
            # else:
            #     h_=h_.reshape([b,c,h*w])

        q_list = []
        k_list = []
        v_list = []
        if self.need_pool and (self.big_n == 0 or self.force_bmm_batch):
            q = q.reshape(b, c, w * h)
            k = k.reshape(b, c, w * h)
            v = v.reshape(b, c, w * h)
            for i in range(head):
                q_s = q[:, :, i::head]
                k_s = k[:, :, i::head]
                v_s = v[:, :, i::head]
                # print("q_s:",q_s.shape)
                q_list.append(q_s)
                k_list.append(k_s)
                v_list.append(v_s)

            # print(q_s.shape)
            q = torch.stack(q_list, dim=1)
            k = torch.stack(k_list, dim=1)
            v = torch.stack(v_list, dim=1)

        elif not self.need_pool:
            # if self.use_MultiHeadSelfAttention:
            #     pass
            # else:
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
        else:
            q = q.view(q.size(0), -1)
            q -= q.min(1, keepdim=True)[0]
            q /= q.max(1, keepdim=True)[0]
            # q = q.view(batch_size, height, width)

            k = k.view(k.size(0), -1)
            k -= k.min(1, keepdim=True)[0]
            k /= k.max(1, keepdim=True)[0]

        # if not self.need_pool:
        #     q = self.small_q(q)
        #     k = self.small_k(k)
        #     v = self.small_v(v)
        # print("after q:",q.shape)

        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.float):
                if self.need_pool:
                    if self.big_n > 0 and not self.force_bmm_batch:
                        # print(h_)
                        if self.svd:
                            h_in = self.cal_bmm_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                        else:
                            h_in = self.cal_bmm_no_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                    else:
                        h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1)

                else:
                    # if not self.use_MultiHeadSelfAttention:
                    h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                    # else:
                    #     h_in = self.small_attention(h_)

            # h_in = h_in.to(torch.float32)
        else:
            if self.need_pool:
                if self.big_n > 0 and not self.force_bmm_batch:
                    if self.svd:
                        h_in = self.cal_bmm_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                    else:
                        h_in = self.cal_bmm_no_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)

                else:
                    h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1)
            else:
                # if not self.use_MultiHeadSelfAttention:
                h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                # else:
                #     h_in = self.small_attention(h_)

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
            # if self.use_MultiHeadSelfAttention:
            #     print(h_in.shape)
            #
            #     h_=h_in
            #     h_ = h_.reshape(b, c, h, w)
            #
            #     print(h_.shape)
            # else:
            h_ = self.small_proj_out(h_in)
            h_ = h_.reshape(b, c, h, w)
        else:
            h_ = self.proj_out(h_in)

        return h_

    def forward(self, x):
        # if self.big_n > 0 and not self.force_bmm_batch:
        #     print("x:",x)
        x_n = self.norm(x)
        # if self.big_n > 0 and not self.force_bmm_batch:
        #     print("x_n:", x_n)
        if not self.need_pool and self.use_MultiHeadSelfAttention:
            b, c, h, w = x_n.shape
            x_n = x_n.reshape([b, c, h * w])
            x_n = self.small_attention(x_n)
            x_n = x_n.reshape([b, c, h, w])
            h_ = x_n
        else:
            if self.mixed_precision:
                # Use autocast to allow operations to run in lower precision
                with autocast(dtype=torch.bfloat16):
                    h_ = self.call_funtion(x_n)
            else:
                h_ = self.call_funtion(x_n)
        # print("attn")

        return x_n + h_


class AttnBlock6_a(nn.Module):
    def __init__(self, in_channels, size, min_head=4, fp16=False, force_bmm_batch=False, test=False,
                 use_MultiHeadSelfAttention=False, svd=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.n = int(size // 32)
        self.big_n = int(size // 64)
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
            if self.big_n > 1:
                self.gap = self.n // 4
                self.max_p = torch.nn.AvgPool1d(kernel_size=self.gap, stride=self.gap)
                    # self.q_dense = torch.nn.Linear(self.size * self.size,
                    #                                16384, )
                    # self.k_dense = torch.nn.Linear(self.size * self.size,
                    #                                16384, )
                    # self.v_dense = torch.nn.Linear(self.size * self.size,
                    #                                16384, )
                    #
                    #
                    # self.v_dense_back = torch.nn.Linear(16384,
                    #                                     self.size * self.size, )

        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0, bias=False)
        if not self.need_pool:
            if self.use_MultiHeadSelfAttention:
                self.small_attention = FlashAttention(dim=size ** 2, heads=min_head, mixed_precision=False)
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

            # n_channels = in_channels // head
            # self.n_channels = n_channels
            # self.small_q = torch.nn.Conv3d(in_channels,
            #                                in_channels,
            #                                kernel_size=1,
            #                                stride=1,
            #                                padding=0)
            #
            # self.small_k = torch.nn.Conv3d(in_channels,
            #                                in_channels,
            #                                kernel_size=1,
            #                                stride=1,
            #                                padding=0)
            # self.small_v = torch.nn.Conv3d(in_channels,
            #                                in_channels,
            #                                kernel_size=1,
            #                                stride=1,
            #                                padding=0)
            self.small_proj_out = torch.nn.Conv3d(head,
                                                  1,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding="same", bias=False)

    def cal_bmm_batch(self, q, k, v, b, c, h, w, d_1, h_1, w_1, hw_1_kv):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # print("q:",q.dtype)
        # print("q:",q.shape)

        q = q.reshape(b * d_1, c, h_1 * w_1)
        q = q.permute(0, 2, 1)  # b,hw,c
        # print("w_in:",w_in.dtype)
        # print("size:",self.size,"q:",q.shape)
        # k = self.avgp(k)
        k = k.reshape(b * d_1, c, hw_1_kv)  # b,c,hw

        # print("k:",k.shape)

        # q.to(torch.float8_e4m3fn)
        # k.to(torch.float8_e4m3fn)
        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print("w_in:",w_in.dtype)
        # w_in=w_in.to(torch.float32)
        w_in = w_in * (int(c) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)
        # w_in.to(torch.float8_e4m3fn)
        # print("w_in:", w_in.shape)
        # attend to values
        v = v.reshape(b * d_1, c, hw_1_kv)
        # print("v:", v.shape)

        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # v.to(torch.float8_e4m3fn)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # print("h_in:", h_in.shape)

        h_in = h_in.reshape(b, d_1, c, h_1 * w_1)
        h_in_list = h_in.split(dim=1, split_size=1)
        # print(q_list)
        # cc = torch.stack(q_list, dim=4).view(2, 3, 16)
        h_in = torch.stack(h_in_list, dim=4).view(b, c, d_1 * h_1 * w_1)
        h_in = h_in.reshape(b, c, h, w)
        # print("bmm b h_in:",h_in.dtype)
        # print("bmm b h_in:",h_in.shape)

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
            h_1 = h // self.n
            w_1 = w // self.n
            #     h_1 = 128
            #     w_1 = 128
            # else:
            # h_1 = h
            # w_1 = w
            hw_1_kv = h_1 * w_1
            if self.big_n > 1:
                hw_1_kv = 32 * (32 // self.gap)


        else:
            # if not self.use_MultiHeadSelfAttention:
            h_1 = h
            w_1 = w
            q = h_
            k = h_
            v = h_
            # else:
            #     h_=h_.reshape([b,c,h*w])

        q_list = []
        k_list = []
        v_list = []
        if self.need_pool:
            # if self.big_n > 1:
            q = q.reshape(b, c, w * h)
            k = k.reshape(b, c, w * h)
            v = v.reshape(b, c, w * h)

                # q = self.q_dense(q)
                # k = self.k_dense(k)
                # k = self.k_dense(k)
            for i in range(head):
                q_s = q[:, :, i::head]
                k_s = k[:, :, i::head]
                v_s = v[:, :, i::head]
                # print("q_s:",q_s.shape)
                if self.big_n>1:
                    k_s=self.max_p(k_s)
                    # print("k_s:",k_s.shape)
                    v_s = self.max_p(v_s)
                    # print("v_s:", v_s.shape)
                q_list.append(q_s)
                k_list.append(k_s)
                v_list.append(v_s)

            q = torch.stack(q_list, dim=1)
            # print("q:",q.shape)
            k = torch.stack(k_list, dim=1)
            # print("k:",k.shape)
            v = torch.stack(v_list, dim=1)

        elif not self.need_pool:
            # if self.use_MultiHeadSelfAttention:
            #     pass
            # else:
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
            with autocast(dtype=torch.float):
                if self.need_pool:
                    # if self.big_n > 0 and not self.force_bmm_batch:
                        # print(h_)
                    #     if self.svd:
                    #         h_in = self.cal_bmm_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                    #     else:
                    #         h_in = self.cal_bmm_no_svd(q, k, v, b, c, h, w, d_1, h_1, w_1)
                    # else:
                    h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1, hw_1_kv)
                    # if self.big_n > 1:
                    #     h_in = self.cal_bmm(q, k, v, b, c, h_1, h_1, d_1, h_1, h_1)
                    # else:
                    #     h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)


                else:
                    # if not self.use_MultiHeadSelfAttention:
                    h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                    # else:
                    #     h_in = self.small_attention(h_)

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
                h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1, hw_1_kv)
                # if self.big_n > 1:
                #     h_in = self.cal_bmm(q, k, v, b, c, h_1, h_1, d_1, h_1, h_1)
                # else:
                #     h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
            else:
                # if not self.use_MultiHeadSelfAttention:
                h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
                # else:
                #     h_in = self.small_attention(h_)

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
            # if self.use_MultiHeadSelfAttention:
            #     print(h_in.shape)
            #
            #     h_=h_in
            #     h_ = h_.reshape(b, c, h, w)
            #
            #     print(h_.shape)
            # else:
            h_ = self.small_proj_out(h_in)
            h_ = h_.reshape(b, c, h, w)
        else:
            # if self.big_n>1:

            h_ = self.proj_out(h_in)

        return h_

    def forward(self, x):
        # if self.big_n > 0 and not self.force_bmm_batch:
        x_n = self.norm(x)
        if not self.need_pool and self.use_MultiHeadSelfAttention:
            b, c, h, w = x_n.shape
            x_n = x_n.reshape([b, c, h * w])
            x_n = self.small_attention(x_n)
            x_n = x_n.reshape([b, c, h, w])
            h_ = x_n
        else:
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


class AttnBlock6_cross(nn.Module):
    def __init__(self, in_channels, size, min_head=4, fp16=False):
        super().__init__()
        self.mixed_precision = fp16
        self.in_channels = in_channels
        self.n = int(size // 32)
        self.size = size
        # self.max_size=256
        # self.max_n=size//self.max_size

        if self.n > 1:
            self.need_pool = True
            head = self.n ** 2
            self.head = head
            # if self.n < 4:
            #     head = min_head
            #     self.head = head
        else:
            self.need_pool = False
            head = min_head
            self.head = head
        # self.num_worker = 4
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        # self.norm_out = nn.GroupNorm(num_groups=1, num_channels=in_channels)

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
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0, bias=False)

        if not self.need_pool:
            self.q_group = [torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda() for i
                            in range(head)]

            self.k_group = [torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda() for i
                            in range(head)]
            self.v_group = [torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            dtype=torch.bfloat16 if self.mixed_precision else torch.float).cuda() for i
                            in range(head)]

        # n_channels = in_channels // head
        # self.n_channels = n_channels
        # self.small_q = torch.nn.Conv3d(in_channels,
        #                                in_channels,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)
        #
        # self.small_k = torch.nn.Conv3d(in_channels,
        #                                in_channels,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)
        # self.small_v = torch.nn.Conv3d(in_channels,
        #                                in_channels,
        #                                kernel_size=1,
        #                                stride=1,
        #                                padding=0)
        self.small_proj_out = torch.nn.Conv3d(head,
                                              1,
                                              kernel_size=1,
                                              stride=1,
                                              padding="same", bias=False)
        # if self.max_n>1:
        #     self.avgp=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        # if self.need_pool:
        #     self.upsample = torch.nn.Upsample(scale_factor=(self.n, self.n), mode='bicubic')

    def cal_bmm_batch(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

        # print("q:",q.dtype)
        # print("q:",q.shape)

        q = q.reshape(b * d_1, c, h_1 * w_1)
        q = q.permute(0, 2, 1)  # b,hw,c
        # print("w_in:",w_in.dtype)
        # print("q:",q.shape)

        k = k.reshape(b * d_1, c, h_1 * w_1)  # b,c,hw
        # print("k:",k.shape)

        # q.to(torch.float8_e4m3fn)
        # k.to(torch.float8_e4m3fn)
        w_in = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # print("w_in:",w_in.dtype)
        # w_in=w_in.to(torch.float32)
        w_in = w_in * (int(c) ** (-0.5))

        w_in = torch.nn.functional.softmax(w_in, dim=2, dtype=torch.float32)
        # w_in.to(torch.float8_e4m3fn)
        # print("w_in:", w_in.shape)
        # attend to values
        v = v.reshape(b * d_1, c, h_1 * w_1)
        # print("v:", v.shape)

        w_in = w_in.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # v.to(torch.float8_e4m3fn)

        h_in = torch.bmm(v, w_in)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # print("h_in:", h_in.shape)

        h_in = h_in.reshape(b, d_1, c, h_1 * w_1)
        h_in_list = h_in.split(dim=1, split_size=1)
        # print(q_list)
        # cc = torch.stack(q_list, dim=4).view(2, 3, 16)
        h_in = torch.stack(h_in_list, dim=4).view(b, c, d_1 * h_1 * w_1)
        h_in = h_in.reshape(b, c, h, w)
        # print("bmm b h_in:",h_in.dtype)
        # print("bmm b h_in:",h_in.shape)

        return h_in

    def cal_bmm(self, q, k, v, b, c, h, w, d_1, h_1, w_1):

        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        if self.mixed_precision:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

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

    def call_funtion(self, q_x, k_x):
        q_h_ = q_x
        k_h_ = k_x

        # compute attention
        b, c, h, w = q_h_.shape
        head = self.head
        # print("head:",head)
        d_1 = head
        c_1 = c // head
        if self.need_pool:
            q = self.q(q_h_)
            k = self.k(k_h_)
            v = self.v(k_h_)
            h_1 = h // self.n
            w_1 = w // self.n
        else:
            h_1 = h
            w_1 = w
            q = q_h_
            k = k_h_
            v = k_h_

        q_list = []
        k_list = []
        v_list = []
        if self.need_pool:
            q = q.reshape(b, c, w * h)
            k = k.reshape(b, c, w * h)
            v = v.reshape(b, c, w * h)
            for i in range(head):
                q_s = q[:, :, i::head]
                k_s = k[:, :, i::head]
                v_s = v[:, :, i::head]
                # print("q_s:",q_s.shape)
                q_list.append(q_s)
                k_list.append(k_s)
                v_list.append(v_s)

            # print(q_s.shape)
            q = torch.stack(q_list, dim=1)
            k = torch.stack(k_list, dim=1)
            v = torch.stack(v_list, dim=1)

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

        # if not self.need_pool:
        #     q = self.small_q(q)
        #     k = self.small_k(k)
        #     v = self.small_v(v)
        # print("after q:",q.shape)

        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.float16 if self.size > 128 else torch.float):
                if self.need_pool:
                    h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1)
                else:
                    h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)
            # h_in = h_in.to(torch.float32)
        else:
            if self.need_pool:
                h_in = self.cal_bmm_batch(q, k, v, b, c, h, w, d_1, h_1, w_1)
            else:
                h_in = self.cal_bmm(q, k, v, b, c, h, w, d_1, h_1, w_1)

        if not self.need_pool:
            h_ = self.small_proj_out(h_in)
            # h_ = torch.mean(h_in, dim=1)
            h_ = self.proj_out(h_)
            # h_ = h_.reshape(b, c, h, w)
        else:
            print(h_in.shape)
            h_ = self.proj_out(h_in)

        return h_

    def forward(self, input_data):
        q, k = input_data[0], input_data[1]
        q_x = self.norm(q)
        k_x = self.norm(k)
        if self.mixed_precision:
            # Use autocast to allow operations to run in lower precision
            with autocast(dtype=torch.bfloat16):
                h_ = self.call_funtion(q_x, k_x)
        else:
            h_ = self.call_funtion(q_x, k_x)
        # print("attn")

        return q + h_


class downsample(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, use_residual=True, need_attention=0,
                 attn_size=128, fp16=False, head=4):
        super().__init__()
        self.use_residual = use_residual
        self.need_attention = need_attention
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 2:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=4, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 1:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=4, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 0:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=4, fp16=fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        self.result = nn.Sequential(*layers)

        # if apply_batchnorm:
        #     self.result = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         nn.GroupNorm(num_groups=1, num_channels=out_channels),
        #         # AttnBlock(out_channels,attn_size) if need_attention else None,
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size) if need_attention else None,
        #         nn.GroupNorm(num_groups=1, num_channels=out_channels),
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size) if need_attention else None,
        #         nn.GroupNorm(num_groups=1, num_channels=out_channels),
        #         nn.LeakyReLU(inplace=True),
        #     )
        # else:
        #     self.result = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         # AttnBlock(out_channels, attn_size) if need_attention else None,
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size) if need_attention else None,
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size) if need_attention else None,
        #         nn.LeakyReLU(inplace=True),
        #     )

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


# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     feat_var = feat.view(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1)
#     feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return feat_mean, feat_std
#
#
# def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std(style_feat)
#     content_mean, content_std = calc_mean_std(content_feat)
#
#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / (content_std.expand(size) + eps)
#     print(f"content_mean :", str(torch.isnan(content_mean).any()))
#     print(f"content_std :", str(torch.isnan(content_std).any()))
#     print(f"normalized_feat :", str(torch.isnan(normalized_feat).any()))
#     print(f"style_mean.expand(size) :", str(torch.isnan(style_mean.expand(size)).any()))
#     print(f"style_std.expand(size):", str(torch.isnan(style_std.expand(size)).any()))
#
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)

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


class upsample(nn.Module):

    def __init__(self, in_channels, out_channels, size, z_dim=512, apply_batchnorm=True, apply_dropout=False,
                 use_residual=True,
                 need_attention=0, use_mapping=False, use_fp16=False, attn_size=128, atten_fp16=False, head=4):
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
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 1:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False))
        if need_attention > 2:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=4, fp16=atten_fp16))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.result = nn.Sequential(*layers)

        # if apply_batchnorm:
        #     self.result = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         nn.GroupNorm(num_groups=1, num_channels=out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size),
        #         nn.GroupNorm(num_groups=1, num_channels=out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size),
        #         nn.GroupNorm(num_groups=1, num_channels=out_channels),
        #         nn.ReLU(inplace=True),
        #     )
        # else:
        #     self.result = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same', bias=False),
        #         AttnBlock3(out_channels, attn_size),
        #         nn.ReLU(inplace=True),
        #     )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        )
        # if need_attention:
        #     self.attn = nn.Sequential(
        #         AttnBlock(out_channels, attn_size)
        #     )
        self.weight = torch.nn.Parameter(
            torch.randn([self.in_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.affine = nn.Sequential(FullyConnectedLayer(z_dim, in_channels, bias_init=1))

    def forward(self, inputs):
        # print(inputs)
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
            # print(x_o.shape)
            # print(x_r.shape)

            x = torch.add(x_o, x_r)
        else:
            x = x_o
        # if self.need_attention:
        #     x = self.attn(x)
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


class Generator(nn.Module):
    def __init__(self, lr_multiplier=0.01, z_dim=512):
        scale = 2
        super(Generator, self).__init__()

        self.down_1 = nn.Sequential(
            downsample(3, 64 // scale, 4, apply_batchnorm=False, use_residual=False, need_attention=1,
                       attn_size=512, fp16=True, head=1))
        self.down_2 = nn.Sequential(
            downsample(64 // scale, 128 // scale, 4, need_attention=1, attn_size=256, fp16=True, head=3))
        self.down_3 = nn.Sequential(
            downsample(128 // scale, 256 // scale, 4, need_attention=2, attn_size=128, fp16=True, head=2))
        self.down_4 = nn.Sequential(
            downsample(256 // scale, 512 // scale, 4, need_attention=2,
                       attn_size=64, fp16=True, head=4))  # (batch_size, 16, 16, 512) 32
        self.down_5 = nn.Sequential(
            downsample(512 // scale, 768 // scale, 4, need_attention=3, attn_size=32,
                       fp16=True, head=4))  # (batch_size, 8, 8, 512) 16
        self.down_6 = nn.Sequential(
            downsample(768 // scale, 1024 // scale, 4, need_attention=3, attn_size=16,
                       fp16=True, head=4))  # (batch_size, 4, 4, 512) 8
        self.down_7 = nn.Sequential(
            downsample(1024 // scale, 1536 // scale, 4, need_attention=3, attn_size=8,
                       fp16=True, head=4))  # (batch_size, 2, 2, 512) 4
        self.down_8 = nn.Sequential(
            downsample(1536 // scale, 2048 // scale, 4, need_attention=3, attn_size=4,
                       fp16=True, head=4))  # (batch_size, 1, 1, 512) 2
        self.down_9 = nn.Sequential(downsample(2048 // scale, 4096 // scale, 2, apply_batchnorm=True, use_residual=True,
                                               need_attention=3, attn_size=2, fp16=True, head=4))  # 1
        # downsample(512, 4),  # (batch_size, 1, 1, 512)

        self.down_stack = [self.down_1, self.down_2, self.down_3, self.down_4, self.down_5, self.down_6, self.down_7,
                           self.down_8, self.down_9]

        self.up_1 = nn.Sequential(
            upsample(4096 // scale, 2048 // scale, 2, apply_dropout=True, need_attention=3, attn_size=2,
                     atten_fp16=True, head=4))  # 2
        self.up_2 = nn.Sequential(
            upsample(4096 // scale, 1536 // scale, 4, apply_dropout=True, need_attention=3, attn_size=4,
                     atten_fp16=True, head=4))  # 4
        self.up_3 = nn.Sequential(upsample(3072 // scale, 1024 // scale, 4, apply_dropout=True,
                                           need_attention=3, attn_size=8,
                                           atten_fp16=True, head=4))  # (batch_size, 2, 2, 1024) 8
        self.up_4 = nn.Sequential(
            upsample(2048 // scale, 768 // scale, 4, need_attention=3, attn_size=16,
                     atten_fp16=True, head=4))  # (batch_size, 4, 4, 1024) 16
        self.up_5 = nn.Sequential(
            upsample(1536 // scale, 512 // scale, 4, need_attention=3, attn_size=32,
                     atten_fp16=True, head=4))  # (batch_size, 8, 8, 1024) 32
        self.up_6 = nn.Sequential(
            upsample(1024 // scale, 256 // scale, 4, need_attention=2,
                     attn_size=64, atten_fp16=True, head=4))  # (batch_size, 16, 16, 1024) 64
        self.up_7 = nn.Sequential(
            upsample(512 // scale, 128 // scale, 4, need_attention=2,
                     attn_size=128, atten_fp16=True, head=2))  # (batch_size, 32, 32, 512) 128
        self.up_8 = nn.Sequential(
            upsample(256 // scale, 64 // scale, 4, need_attention=1,
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

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=1, padding="same", bias=False),
        ]
        if need_attention:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=2, fp16=atten_fp16, force_bmm_batch=False))
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


class nodownsample_d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True, need_attention=True, attn_size=128):
        super().__init__()
        self.need_attention = need_attention
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=1, padding="same", bias=False),
        ]
        if need_attention:
            layers.append(AttnBlock6_a(out_channels, attn_size, min_head=1))
        if apply_batchnorm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        layers.append(nn.LeakyReLU(inplace=True))

        self.result = nn.Sequential(*layers)
        # if need_attention:
        #     self.attn = nn.Sequential(
        #         AttnBlock(out_channels, attn_size)
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
