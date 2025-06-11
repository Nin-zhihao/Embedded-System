import itertools
import os
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import random


class LimitedStepsCycleDataset(IterableDataset):
    def __init__(self, finite_dataset, steps=None):
        """
        从有限数据集中创建一个有限步骤的无限循环数据集。

        参数：
            finite_dataset (Dataset): 一个有限的PyTorch数据集实例。
            steps (int, optional): 指定在无限重复中允许的最大步骤数。默认为 None，表示无限。
        """
        super().__init__()
        self.finite_dataset = finite_dataset
        self.steps = steps

    def set_steps(self, steps):
        """
        设置步骤限制。

        参数：
            steps (int): 要设置的步骤数。
        """
        self.steps = steps

    def __iter__(self):
        # 使用 itertools.cycle 实现无限循环
        cycle_iter = itertools.cycle(self.finite_dataset)
        step = 0
        while self.steps is None or step < self.steps:
            try:
                yield next(iter(cycle_iter))
                step += 1
            except StopIteration:
                # 通常不会发生，因为 cycle 是无限的，但为了保险起见
                cycle_iter = itertools.cycle(self.finite_dataset)


class ImagePathDataset_sl(Dataset):
    def __init__(self, image_path, image_size=(512, 512),to_normal=True):
        super().__init__()
        # self.n_steps = n_steps  # defines an epoch -> thus how often we check the callbacks, compute the val_loss, etc.
        # self.current_file = 0
        # self.step=step
        image_paths=get_image_paths_from_dir(image_path)
        self.image_size = image_size
        self.image_paths = image_paths
        self.n_files = len(image_paths)
        self._length = len(image_paths)
        # if shuffle:
        #     random.shuffle(image_paths)
        self.to_normal = to_normal  # 是否归一化到[-1, 1]
        style_image = Image.open("./style_img_raw.jpg")
        style_image = np.array(style_image, dtype=np.uint8)  # H*W*C
        self.transform_s = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        style_image = self.transform_s(style_image)


        style_image = (style_image *2 ) - 1
        self.style_image = style_image


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # p = 0.0
        # if index >= self.current_file:
        # if index >= self._length:
        # print(self.current_file)
        # index = self.current_file
        try:
            # print(index)
            # p = 1.0

            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            img_path = self.image_paths[index]
            # print(img_path)
            image = None
            try:
                image = Image.open(img_path)
            except BaseException as e:
                print(img_path)

            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image = np.array(image, dtype=np.uint8)  # H*W*C

            w = image.shape[1]
            w = w // 5
            # style_image = image[:, 2 * w:, :]
            # input_image = image[:, w:2 * w, :]
            real_image = image[:, :w, :]
            # style_img = img_label[:, 0:w, :]
            input_image_b = image[:, w:2 * w, :]
            input_image_lh = image[:, 2 * w:3 * w, :]
            input_image_rh = image[:, 3 * w:4 * w, :]
            input_image_f = image[:, 4 * w:5 * w, :]

            real_image = self.transform_s(real_image)
            input_image_b = transform(input_image_b)
            input_image_lh = transform(input_image_lh)
            input_image_rh = transform(input_image_rh)
            input_image_f = transform(input_image_f)
            # real_image = torch.from_numpy(real_image)
            # input_image_b = torch.from_numpy(input_image_b)
            # input_image_lh = torch.from_numpy(input_image_lh)
            # input_image_rh = torch.from_numpy(input_image_rh)
            # input_image_f = torch.from_numpy(input_image_f)
            #
            # real_image = nnf.interpolate(real_image, size=(224, 224), mode='bicubic', align_corners=False)
            # input_image_b = nnf.interpolate(input_image_b, size=(224, 224), mode='bicubic', align_corners=False)
            # input_image_lh = nnf.interpolate(input_image_lh, size=(224, 224), mode='bicubic', align_corners=False)
            # input_image_rh = nnf.interpolate(input_image_rh, size=(224, 224), mode='bicubic', align_corners=False)
            # input_image_f = nnf.interpolate(input_image_f, size=(224, 224), mode='bicubic', align_corners=False)

            if self.to_normal:
                input_image_b = (input_image_b *2 ) - 1
                input_image_lh = (input_image_lh *2 ) - 1
                input_image_rh = (input_image_rh *2 ) - 1
                input_image_f = (input_image_f *2 ) - 1
                real_image = (real_image *2 ) - 1

                # real_image = (real_image - 0.5) * 2.
                # real_image.clamp_(-1., 1.)
                # input_image_b = (input_image_b - 0.5) * 2.
                # input_image_b.clamp_(-1., 1.)
                # input_image_lh = (input_image_lh - 0.5) * 2.
                # input_image_lh.clamp_(-1., 1.)
                # input_image_rh = (input_image_rh - 0.5) * 2.
                # input_image_rh.clamp_(-1., 1.)
                # input_image_f = (input_image_f - 0.5) * 2.
                # input_image_f.clamp_(-1., 1.)

            input_image = torch.cat((input_image_b, input_image_lh, input_image_rh, input_image_f), 0)
            style_image = self.style_image
            return input_image, real_image, style_image, img_path
        except Exception as msg:
            print(msg)
            traceback.print_exc()
        # print(self.style_image.shape)
        # print(input_image_b.shape)
        #
        # print(image.shape)
        # image_name = Path(img_path).stem
        # print(torch.max(style_image))
        # print(torch.min(style_image))




def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            if "jpg" in fpath or "jpeg" in fpath:
                image_paths.append(fpath)
    return image_paths