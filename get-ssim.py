#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute PSNR / SSIM / UCIQE / LPIPS between two folders.
- 文件夹A: 增强结果（预测）
- 文件夹B: 参考/GT
按“文件名去扩展名后的 stem”进行配对，例如 a001.png 与 a001.jpg 会被认为是一对。
依赖：torch, torchvision, torchmetrics
可选依赖：
  - metrics.uciqe.batch_uciqe（若没有则跳过UCIQE）
  - lpips（richzhang 实现，若没有则跳过LPIPS）
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

# =============== 直接在这里设置路径与开关 ===============
ENHANCED_DIR = "/root/sqwu/UIR/output/UFO-ours-7-500"      # 增强图像文件夹
REFERENCE_DIR = "/root/sqwu/data/UFO-120/test/target"     # 参考/GT 文件夹
OUT_CSV = "metrics_results.csv"  # 导出CSV路径
STRICT_SIZE = False              # True：若尺寸不一致则报错；False：将增强图resize到GT尺寸
DEVICE = "cuda"                  # "cuda" 或 "cpu"；若无GPU则自动回退到CPU
LPIPS_BACKBONE = "alex"          # "alex" | "vgg" | "squeeze"
# =====================================================

# 尝试导入UCIQE（可选）
try:
    from metrics.uciqe import batch_uciqe  # 期望签名：batch_uciqe(imgs[N,3,H,W] in [0,1])
    HAS_UCIQE = True
except Exception as e:
    batch_uciqe = None
    HAS_UCIQE = False
    print("[WARN] 未能导入 metrics.uciqe.batch_uciqe，将跳过UCIQE计算：", e, file=sys.stderr)

# 尝试导入LPIPS（可选）
try:
    import lpips as _lpips  # pip install lpips
    HAS_LPIPS = True
except Exception as e:
    _lpips = None
    HAS_LPIPS = False
    print("[WARN] 未能导入 lpips，将跳过LPIPS计算：", e, file=sys.stderr)


def collect_images(folder: Path, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')) -> Dict[str, Path]:
    """遍历收集图片：返回 {stem: 路径}"""
    mapping = {}
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.stem] = p
    return mapping


def load_image_as_float01(path: Path) -> torch.Tensor:
    """
    读图为 float32 [3,H,W] 且范围 [0,1]
    - 灰度图复制到3通道
    - RGBA丢弃Alpha
    """
    img = read_image(str(path))  # uint8 [C,H,W]
    if img.ndim != 3:
        raise ValueError(f"Unexpected tensor shape from {path}: {img.shape}")
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3, ...]
    elif img.shape[0] != 3:
        raise ValueError(f"Unsupported channel count {img.shape[0]} for {path}")
    img = img.float() / 255.0
    return img.clamp(0, 1)


def main():
    A = Path(ENHANCED_DIR)
    B = Path(REFERENCE_DIR)
    assert A.is_dir(), f"增强目录不存在: {A}"
    assert B.is_dir(), f"参考目录不存在: {B}"

    mapA = collect_images(A)
    mapB = collect_images(B)
    common = sorted(set(mapA.keys()) & set(mapB.keys()))
    missing_A = sorted(set(mapB.keys()) - set(mapA.keys()))
    missing_B = sorted(set(mapA.keys()) - set(mapB.keys()))

    if len(common) == 0:
        print("[ERROR] A 与 B 无匹配的文件名（去扩展名后）。", file=sys.stderr)
        print(f"A count={len(mapA)}, B count={len(mapB)}", file=sys.stderr)
        sys.exit(1)

    if missing_A:
        print(f"[WARN] B中有 {len(missing_A)} 个文件在A中缺失（按stem匹配），示例：{missing_A[:5]}")
    if missing_B:
        print(f"[WARN] A中有 {len(missing_B)} 个文件在B中缺失（按stem匹配），示例：{missing_B[:5]}")

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")

    # 预建 LPIPS 模型（若可用）
    lpips_model = None
    if HAS_LPIPS:
        try:
            lpips_model = _lpips.LPIPS(net=LPIPS_BACKBONE).to(device)
            lpips_model.eval()
        except Exception as e:
            print("[WARN] 初始化 LPIPS 失败，将跳过LPIPS计算：", e, file=sys.stderr)
            lpips_model = None

    rows: List[Tuple[str, float, float, float, float]] = []  # (stem, psnr, ssim, uciqe, lpips)
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_uciqe = 0.0
    n_uciqe = 0
    sum_lpips = 0.0
    n_lpips = 0

    for stem in tqdm(common, desc="Evaluating"):
        pa = mapA[stem]
        pb = mapB[stem]

        pred = load_image_as_float01(pa)  # [3,H,W]
        gt   = load_image_as_float01(pb)  # [3,H,W]

        if STRICT_SIZE:
            if pred.shape[1:] != gt.shape[1:]:
                raise AssertionError(f"Size mismatch for {stem}: pred={list(pred.shape[1:])}, gt={list(gt.shape[1:])}")
        else:
            if pred.shape[1:] != gt.shape[1:]:
                pred = resize(pred, list(gt.shape[1:]), antialias=True)

        pred_b = pred.unsqueeze(0).to(device)
        gt_b   = gt.unsqueeze(0).to(device)

        with torch.no_grad():
            # PSNR / SSIM（参考图像质量指标）
            psnr = float(peak_signal_noise_ratio(pred_b, gt_b, data_range=1.0).item())
            ssim = float(structural_similarity_index_measure(pred_b, gt_b, data_range=1.0).item())

            # UCIQE（无参考图像质量，通常对增强结果单边评估）
            if HAS_UCIQE:
                try:
                    u = batch_uciqe(pred_b)  # 常见实现：对增强结果计算画质指标
                    u = float(u.item() if hasattr(u, "item") else float(u))
                except Exception as e:
                    print(f"[WARN] UCIQE 计算失败（{stem}）：{e}", file=sys.stderr)
                    u = float("nan")
            else:
                u = float("nan")

            # LPIPS（感知距离，越小越好）
            if lpips_model is not None:
                try:
                    # lpips 期望输入范围为 [-1, 1]
                    pred_n11 = pred_b * 2 - 1
                    gt_n11   = gt_b * 2 - 1
                    l = lpips_model(pred_n11, gt_n11)
                    l = float(l.item() if hasattr(l, "item") else float(l))
                except Exception as e:
                    print(f"[WARN] LPIPS 计算失败（{stem}）：{e}", file=sys.stderr)
                    l = float("nan")
            else:
                l = float("nan")

        rows.append((stem, psnr, ssim, u, l))
        sum_psnr += psnr
        sum_ssim += ssim
        if not (u != u):  # 非 NaN
            sum_uciqe += u
            n_uciqe += 1
        if not (l != l):  # 非 NaN
            sum_lpips += l
            n_lpips += 1

    avg_psnr = sum_psnr / len(common)
    avg_ssim = sum_ssim / len(common)
    avg_uciqe = (sum_uciqe / n_uciqe) if n_uciqe > 0 else float("nan")
    avg_lpips = (sum_lpips / n_lpips) if n_lpips > 0 else float("nan")

    # 写出 CSV
    import csv
    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "psnr", "ssim", "uciqe", "lpips"])
        for r in rows:
            writer.writerow(r)
        writer.writerow([])
        writer.writerow(["AVG", avg_psnr, avg_ssim, avg_uciqe, avg_lpips])

    # 打印汇总
    print("\n===== Summary =====")
    print(f"Pairs evaluated: {len(common)}")
    print(f"Average PSNR:   {avg_psnr:.4f}")
    print(f"Average SSIM:   {avg_ssim:.4f}")
    if n_uciqe > 0:
        print(f"Average UCIQE:  {avg_uciqe:.4f}")
    else:
        print("Average UCIQE:  skipped（未找到 metrics.uciqe.batch_uciqe 或计算失败）")
    if n_lpips > 0:
        print(f"Average LPIPS:  {avg_lpips:.4f}")
    else:
        print("Average LPIPS:  skipped（未安装/初始化失败）")
    print(f"Per-image CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
