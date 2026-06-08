import json
import warnings
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl as kl_ops  # ==== 修改：引入 KL 计算 ====
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from config import Config
from data import get_data
from metrics.uciqe import batch_uciqe
from models.model_ours_2 import MaPU
from utils import *

warnings.filterwarnings('ignore')


def train():
    # Load config
    opt = Config('config_ufo.yml')
    seed_everything(opt.OPTIM.SEED)

    # ==== 修改：根据配置决定是否接入 wandb ====
    accelerator = Accelerator(log_with='wandb') if getattr(opt.OPTIM, "WANDB", False) else Accelerator()
    
    # Ensure save directories exist
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
        os.makedirs(opt.LOG.LOG_DIR if opt.LOG.LOG_DIR else '.', exist_ok=True)
    
    device = accelerator.device

    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("UW", config=config)

    # -------------------
    # Data Loader
    # -------------------
    train_dataset = get_data(
        opt.TRAINING.TRAIN_DIR, 
        opt.MODEL.INPUT, 
        opt.MODEL.TARGET, 
        'train', 
        opt.TRAINING.ORI,
        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H}
    )
    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=True,          # 使用 shuffle 而非 sampler
        num_workers=16,
        drop_last=False,
        pin_memory=True
    )

    val_dataset = get_data(
        opt.TRAINING.VAL_DIR, 
        opt.MODEL.INPUT, 
        opt.MODEL.TARGET, 
        'test', 
        opt.TRAINING.ORI,
        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H}
    )
    testloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )

    # -------------------
    # Model & Loss
    # -------------------
    model = MaPU()  # ==== 修改：模型已支持训练/测试两阶段 ====
    criterion_recon = nn.SmoothL1Loss()  # 作为重建项（与 PSNR 正相关的度量）

    # ==== 修改：从配置读取 KL 权重，若没有则设置默认值 ====
    kl_weight = float(getattr(opt.OPTIM, "KL_W", 1e-3))
    # （可选）感知损失权重、SSIM/UCiqe 权重也可从配置读取
    w_ssim = float(getattr(opt.OPTIM, "W_SSIM", 0.2))
    w_uciqe = float(getattr(opt.OPTIM, "W_UCIQE", 0.01))

    # -------------------
    # Optimizer & Scheduler
    # -------------------
    optimizer_b = optim.AdamW(
        model.parameters(), 
        lr=opt.OPTIM.LR_INITIAL, 
        betas=(0.9, 0.999), 
        eps=1e-8
    )
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, 
        opt.OPTIM.NUM_EPOCHS, 
        eta_min=opt.OPTIM.LR_MIN
    )

    start_epoch = 1

    # Prepare accelerator
    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    best_psnr_epoch = 1
    best_psnr = 0.0
    size = len(testloader)

    # -------------------
    # Training loop
    # -------------------
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        # ==== 修改：训练阶段走 CVAE 路径，计算 KL + 重建 + 感知指标损失 ====
        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp = data[0].contiguous().to(device, non_blocking=True)  # 放到当前设备
            tar = data[1].to(device, non_blocking=True)

            optimizer_b.zero_grad()

            # ==== 修改：训练前向 —— 返回输出与四个分布 ====
            out, pr_u_dist, pr_s_dist, po_u_dist, po_s_dist = model(inp, tar, True)

            # 重建项（L1/SmoothL1）
            loss_recon = criterion_recon(out, tar)

            # 感知类指标作为辅助损失（注意保持 data_range=1）
            loss_ssim = 1.0 - structural_similarity_index_measure(out, tar, data_range=1.0)
            loss_uciqe = 1.0 - batch_uciqe(out)

            # KL（对 u、s 两组分布分别计算，再取均值）
            kl_u = kl_ops.kl_divergence(po_u_dist, pr_u_dist).mean()
            kl_s = kl_ops.kl_divergence(po_s_dist, pr_s_dist).mean()
            loss_kl = (kl_u + kl_s)

            # 总损失（ELBO 风格）：重建 + λ1·SSIM + λ2·UCIQE + λ3·KL
            train_loss = loss_recon + w_ssim * loss_ssim + w_uciqe * loss_uciqe + kl_weight * loss_kl

            # 反传 & 更新
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # -------------------
        # Validation
        # -------------------
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr_sum, ssim_sum, uciqe_sum = 0.0, 0.0, 0.0

            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                inp = data[0].contiguous().to(device, non_blocking=True)
                tar = data[1].to(device, non_blocking=True)

                with torch.no_grad():
                    # ==== 修改：测试阶段走 MP —— 仅先验、取均值点 ====
                    res = model(inp, None,False)

                # ==== 修改：gather 之后再做度量，避免多卡统计不一致 ====
                res, tar = accelerator.gather((res, tar))

                # 注意：下面的 metrics 会在 CPU 上转 float 以便 .item()
                psnr_sum += peak_signal_noise_ratio(res, tar, data_range=1.0).item()
                ssim_sum += structural_similarity_index_measure(res, tar, data_range=1.0).item()
                # UCIQE 返回张量，这里用 .mean().item() 保守求值
                uciqe_sum += batch_uciqe(res).mean().item()

            psnr = psnr_sum / size
            ssim = ssim_sum / size
            uciqe = uciqe_sum / size

            # Save best model（仅主进程）
            if accelerator.is_local_main_process and psnr > best_psnr:
                best_psnr = psnr
                best_psnr_epoch = epoch
                save_checkpoint(
                    {
                        'state_dict': accelerator.unwrap_model(model).state_dict(),  # ==== 修改：unwrap 模型 ====
                        'epoch': epoch,
                        'best_psnr': best_psnr
                    },
                    epoch,
                    opt.MODEL.SESSION,
                    opt.TRAINING.SAVE_DIR
                )

            # Logging
            accelerator.log({"PSNR": psnr, "SSIM": ssim, "UCIQE": uciqe}, step=epoch)

            if accelerator.is_local_main_process:
                log_stats = {
                    "epoch": epoch,
                    "PSNR": round(psnr, 4),
                    "SSIM": round(ssim, 4),
                    "UCIQE": round(uciqe, 4),
                    "best_PSNR": round(best_psnr, 4),
                    "best_epoch": best_psnr_epoch
                }
                print("epoch: {}, PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, best PSNR: {:.4f}, best epoch: {}"
                      .format(epoch, psnr, ssim, uciqe, best_psnr, best_psnr_epoch))

                log_file_path = os.path.join(opt.LOG.LOG_DIR if opt.LOG.LOG_DIR else '.', opt.TRAINING.LOG_FILE)
                with open(log_file_path, mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats, ensure_ascii=False) + '\n')

    accelerator.end_training()


if __name__ == '__main__':
    train()
