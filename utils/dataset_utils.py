import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        # 直接扫描训练目录，所有图片作为清晰图（用于合成噪声训练）
        name_list = [f for f in os.listdir(self.args.denoise_dir) if f.endswith(('.jpg', '.bmp', '.png'))]
        clean_ids = [self.args.denoise_dir + f for f in name_list]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        # 直接从预先筛选好的索引文件加载
        ref_file = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids = []
        with open(ref_file, "r") as f:
            for line in f:
                id_ = line.strip()
                if id_:
                    temp_ids.append(self.args.dehaze_dir + id_)
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:  # 去噪任务
            clean_id = sample["clean_id"]
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)
            clean_name = clean_id.split("/")[-1].split('.')[0]
            clean_patch = random_augmentation(clean_patch)[0]
            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        elif de_id == 3:  # 去雨
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        elif de_id == 4:  # 去雾
            degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = self._get_nonhazy_name(sample["clean_id"])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return [clean_name, de_id], degrad_patch, clean_patch
    
    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        print(len(self.sample_ids))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        # RealRain-1k 的 gt 文件名与 rainy 完全相同，直接替换目录名
        return rainy_name.replace('rainy', 'gt')

    def _get_nonhazy_name(self, hazy_name):
        # 清晰图与雾图同名，直接替换目录前缀 (适用于 LMHaze 等命名一致的数据集)
        return hazy_name.replace("synthetic", "original")