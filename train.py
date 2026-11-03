import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt


class PromptIRModel(pl.LightningModule):
    def __init__(self, use_ppe=True, use_agf=True):
        super().__init__()
        self.net = PromptIR(decoder=True, use_ppe=use_ppe, use_agf=use_agf)
        self.loss_fn = nn.L1Loss()
        self.use_ppe = use_ppe
        self.use_agf = use_agf

    def forward(self, x):
        return self.net(x)[0]

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, reliability = self.net(degrad_patch)
        loss_l1 = self.loss_fn(restored, clean_patch)

        # ===== 物理损失 =====
        loss_physics = 0.0
        if opt.use_physics_loss:
            T, A, S = self.net.ppe(degrad_patch)
            degrad_gray = 0.299 * degrad_patch[:, 0:1] + 0.587 * degrad_patch[:, 1:2] + 0.114 * degrad_patch[:, 2:3]
            degrad_gray = degrad_gray / (degrad_gray.max() + 1e-8)
            loss_T = F.mse_loss(T, 1.0 - degrad_gray)
            loss_S = torch.mean(torch.abs(S))
            dark = torch.min(restored, dim=1, keepdim=True)[0]
            dark = F.avg_pool2d(dark, 15, stride=1, padding=7)
            loss_dark = dark.mean()
            loss_physics = 0.3 * loss_T + 0.1 * loss_S + 0.1 * loss_dark
            self.log("loss_T", loss_T)
            self.log("loss_S", loss_S)
            self.log("loss_dark", loss_dark)

        loss = loss_l1 + loss_physics
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_l1", loss_l1)
        return loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, reliability = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        ppe_params = [p for n, p in self.net.named_parameters() if 'ppe' in n]
        base_params = [p for n, p in self.net.named_parameters() if 'ppe' not in n]
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': 2e-4},
            {'params': ppe_params, 'lr': 1e-6},
        ])
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=150)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)


def main():
    print("Options")
    print(opt)

    trainset = PromptTrainDataset(opt)
    val_size = int(0.1 * len(trainset))
    train_size = len(trainset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
    trainloader = DataLoader(train_subset, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    valloader = DataLoader(val_subset, batch_size=opt.batch_size, shuffle=False,
                           num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    model = PromptIRModel(use_ppe=opt.use_ppe, use_agf=opt.use_agf)

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="promptir-{epoch:02d}-{val_loss:.4f}",
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
