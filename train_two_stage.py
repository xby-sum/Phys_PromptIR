
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from torchvision import transforms
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
from net.model import PromptIR
from utils.dataset_utils import PromptTrainDataset

from torch.utils.data import Dataset
from PIL import Image
import os
import random


class PairDataset(Dataset):
    """通用配对数据集，支持去噪（noisy/gt）和去雾（hazy/GT）"""
    def __init__(self, root_dir, degraded_subdir='hazy', clean_subdir='GT', patch_size=64):
        self.root_dir = root_dir
        self.degraded_dir = os.path.join(root_dir, degraded_subdir)
        self.clean_dir = os.path.join(root_dir, clean_subdir)
        self.files = sorted([f for f in os.listdir(self.degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.patch_size = patch_size
        print(f"加载配对数据: {len(self.files)} 张 (退化: {degraded_subdir}, 清晰: {clean_subdir})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import torch
        from torchvision import transforms
        fname = self.files[idx]
        degraded_path = os.path.join(self.degraded_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)
        degraded = Image.open(degraded_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        i, j, h, w = transforms.RandomCrop.get_params(degraded, (self.patch_size, self.patch_size))
        degraded = transforms.functional.crop(degraded, i, j, h, w)
        clean = transforms.functional.crop(clean, i, j, h, w)
        degraded = transforms.ToTensor()(degraded)
        clean = transforms.ToTensor()(clean)
        return [fname, 0], degraded, clean

class DenoisePairDataset(Dataset):
    """用于真实噪声数据（SIDD等）的配对数据集，要求目录结构为 gt/ 和 noisy/"""
    def __init__(self, root_dir, patch_size=64):
        self.root_dir = root_dir
        self.gt_dir = os.path.join(root_dir, 'gt')
        self.noisy_dir = os.path.join(root_dir, 'noisy')
        self.files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.patch_size = patch_size
        print(f"加载去噪配对数据: {len(self.files)} 张")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt_path = os.path.join(self.gt_dir, fname)
        noisy_path = os.path.join(self.noisy_dir, fname)
        gt = Image.open(gt_path).convert('RGB')
        noisy = Image.open(noisy_path).convert('RGB')
        i, j, h, w = transforms.RandomCrop.get_params(gt, (self.patch_size, self.patch_size))
        gt = transforms.functional.crop(gt, i, j, h, w)
        noisy = transforms.functional.crop(noisy, i, j, h, w)
        gt = transforms.ToTensor()(gt)
        noisy = transforms.ToTensor()(noisy)
        return [fname, 0], noisy, gt


class TwoStageFinetuneModel(pl.LightningModule):
    def __init__(self, base_ckpt_path=None, freeze_stage=1, use_ppe=True, use_agf=True):
        super().__init__()
        self.net = PromptIR(decoder=True, use_ppe=use_ppe, use_agf=use_agf)
        self.loss_fn = nn.L1Loss()
        self.freeze_stage = freeze_stage
        self.use_ppe = use_ppe
        self.use_agf = use_agf

        if base_ckpt_path and os.path.exists(base_ckpt_path):
            ckpt = torch.load(base_ckpt_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            new_sd = {k[4:] if k.startswith('net.') else k: v for k, v in state_dict.items()}
            self.net.load_state_dict(new_sd, strict=False)
            print(f"✅ 加载预训练权重: {base_ckpt_path}")

        self._apply_freeze(freeze_stage)

    def _apply_freeze(self, stage):
        if stage == 1:
            for name, param in self.net.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("🔒 阶段1：冻结主干，仅适配器可训练")
        elif stage == 2:
            for param in self.net.parameters():
                param.requires_grad = True
            print("🔓 阶段2：全部参数可训练")
        else:
            for param in self.net.parameters():
                param.requires_grad = True
            print("🧪 正常训练（无冻结）")

    def forward(self, x):
        return self.net(x)[0]

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, reliability = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, reliability = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.freeze_stage == 1:
            lr = 1e-4
            params = [p for n, p in self.net.named_parameters() if 'adapter' in n]
        else:
            lr = 1e-5
            params = self.net.parameters()
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=30)
        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    
    # ---- 数据集初始化（优先检测配对数据） ----
    is_pair_data = False
    
    # 去雾检测
    if opt.de_type[0].startswith('dehaze'):
        hazy_path = os.path.join(opt.dehaze_dir, 'hazy')
        gt_path = os.path.join(opt.dehaze_dir, 'GT')
        if os.path.exists(hazy_path) and os.path.exists(gt_path):
            is_pair_data = True
            full_dataset = PairDataset(opt.dehaze_dir, degraded_subdir='hazy', clean_subdir='GT', patch_size=opt.patch_size)
            print(f"✅ 使用去雾配对数据集（hazy/GT），总样本数: {len(full_dataset)}")
    
    # 去噪检测
    if not is_pair_data and opt.de_type[0].startswith('denoise'):
        noisy_path = os.path.join(opt.denoise_dir, 'noisy')
        gt_path = os.path.join(opt.denoise_dir, 'gt')
        if os.path.exists(noisy_path) and os.path.exists(gt_path):
            is_pair_data = True
            full_dataset = PairDataset(opt.denoise_dir, degraded_subdir='noisy', clean_subdir='gt', patch_size=opt.patch_size)
            print(f"✅ 使用去噪配对数据集（noisy/gt），总样本数: {len(full_dataset)}")
    
    if not is_pair_data:
        full_dataset = PromptTrainDataset(opt)
        print(f"使用常规数据集，总样本数: {len(full_dataset)}")

    # 子集采样（支持 subset_size）
    subset_size = getattr(opt, 'subset_size', None)
    if subset_size and subset_size < len(full_dataset):
        import random
        indices = list(range(len(full_dataset)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:subset_size]
        dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"📊 使用子集大小: {subset_size} ({len(dataset)} 张)")
    else:
        dataset = full_dataset
        print(f"📊 使用全部数据: {len(dataset)} 张")

    # 划分训练/验证集
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    valloader = DataLoader(val_subset, batch_size=opt.batch_size, shuffle=False,
                           num_workers=opt.num_workers, pin_memory=True, drop_last=False)


    model = TwoStageFinetuneModel(
        base_ckpt_path=opt.ckpt_path,
        freeze_stage=getattr(opt, 'freeze_stage', 1),
        use_ppe=getattr(opt, 'use_ppe', True),
        use_agf=getattr(opt, 'use_agf', True)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename=f"stage{getattr(opt, 'freeze_stage', 1)}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        callbacks=[early_stop, checkpoint_callback],
        precision="16-mixed",
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=False,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()
