import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
from PIL.Image import Image
from torchvision import transforms
import torch.utils.data
import PIL
import re
import random


class AllWeather:
    def __init__(self, config):
        self.config = config
        # self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.transforms = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])

    def get_loaders(self, parse_patches=True, validation='allweather'):
        train_path = os.path.join(self.config.data.data_dir, 'train/')
        val_path = os.path.join(self.config.data.data_dir, 'test/')

        train_dataset = AllWeatherDataset(train_path,
                                  n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  transforms=self.transforms,
                                  parse_patches=parse_patches)
        val_dataset = AllWeatherDataset(val_path,
                                n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                transforms=self.transforms,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # 训练数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        # 评估数据
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def get_test_loaders(self, parse_patches=True, validation='allWeather'):
        test_path = os.path.join(self.config.data.data_dir, 'test/')

        test_dataset = AllWeatherDataset(test_path,
                                     n=self.config.training.patch_n,
                                     patch_size=self.config.data.image_size,
                                     transforms=self.transforms,
                                     parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.sampling.batch_size,
                                                  shuffle=False,num_workers=self.config.data.num_workers,
                                                  pin_memory=True)
        return test_loader

class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()

        self.dir = dir  # 数据的路径
        input_names = os.listdir(dir+'input')
        gt_names = os.listdir(dir+'target')

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size    # 分patch的大小
        self.transforms = transforms    # 数据处理
        self.n = n                      # patch的数量
        self.parse_patches = parse_patches  # 指示是否对图像进行分块采样

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        # 判断如果输入图像的尺寸已经等于目标输出尺寸,则不需要进行裁剪,直接返回全图的起始位置和尺寸.
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]  # 生成n个高度方向上的随机起始位置,保证裁剪后的图像高度不超过输入图像的高度.
        j_list = [random.randint(0, w - tw) for _ in range(n)]  # 生成n个宽度方向上的随机起始位置,保证裁剪后的图像宽度不超过输入图像的宽度.
        return i_list, j_list, th, tw

    # 根据传入的图像,裁剪图像的大小,裁剪图像的x和y起始位置,进行裁剪.
    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        # 从输入图像文件名中提取图像ID.它通过使用斜杠/来拆分路径,并获取最后一个元素,再通过[:-4]切片删除文件扩展名(通常为.png或.jpg等),从而得到图像ID.
        img_id = re.split('/', input_name)[-1][:-4]
        # 通过PIL.Image.open方法打开输入图像文件.如果self.dir不为None,则从指定的目录加载图像文件,否则从当前工作目录加载图像文件.
        input_img = PIL.Image.open(os.path.join(self.dir, 'input', input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')
        # 判断是否需要进行分块采样
        if self.parse_patches:
            # i和j分别为n个高度方向上的随机起始位置,n个宽度方向上的随机起始位置.h和w分别为output_size的h和w.
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)  # input_img和gt_img是根据随机生成的裁剪参数i和j裁剪出的patch_size大小的图组成的元组.
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            # 对每个图像块进行数据转换,并通过torch.cat函数将输入图像块和对应的标签图像块按通道维度拼接起来,得到多个采样的输入-标签图像对,[6, h, w].
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            # 将采样得到的多个输入-标签图像对按批次维度堆叠,并返回与图像ID对应的元组.
            return torch.stack(outputs, dim=0), img_id
        else:   # 即进行整体采样
            # 对输入图像和标签图像进行调整大小,使其成为16的倍数,以适应整体图像恢复.
            wd_new, ht_new = input_img.size
            # 如果ht_new（图像高度）大于wd_new（图像宽度）且ht_new大于1024像素, 则将wd_new调整为wd_new * 1024 / ht_new,ht_new调整为1024像素.
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024   # 这是为了保持图像的高宽比, 并限制图像高度为1024像素, 以适应较大的高度, 并避免图像变得过于长
            # 如果ht_new小于等于wd_new且wd_new大于1024像素, 则将ht_new调整为ht_new * 1024 / wd_new, wd_new调整为1024像素.
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024   # 这是为了保持图像的高宽比,并限制图像宽度为1024像素,以适应较大的宽度,并避免图像变得过于宽.
            # 将wd_new和ht_new分别调整为最接近16的倍数
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            # 使用PIL.Image.resize()方法将输入图像和标签图像调整为新的尺寸wd_new和ht_new,采用了PIL.Image.ANTIALIAS插值方法进行调整,以保持图像质量
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            # 前3个通道是有雾图像RGB, 后3个通道是干净图像.
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)