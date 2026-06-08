#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算文件夹中所有图像的平均 UIQM 值（带默认参数）
"""

import os
import math
import argparse
from glob import glob
from typing import List
import numpy as np
from PIL import Image
from scipy import ndimage

# ==================== UIQM 模块（保持不变） ====================

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = (1/(K-T_a_L-T_a_R))
    val = sum(x[int(T_a_L+1):int(K-T_a_R)])
    return weight * val

def s_a(x, mu):
    val = 0.0
    for pixel in x:
        val += (pixel - mu) ** 2
    return val / len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R - G
    YB = ((R + G) / 2.0) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt(mu_a_RG**2 + mu_a_YB**2)
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)

def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    m = np.max(mag)
    if m > 0:
        mag *= 255.0 / m
    return mag

def eme(x, window_size):
    k1 = int(x.shape[1] / window_size)
    k2 = int(x.shape[0] / window_size)
    if k1 == 0 or k2 == 0:
        return 0.0
    w = 2.0 / (k1 * k2)
    x = x[:window_size*k2, :window_size*k1]
    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:(k+1)*window_size, l*window_size:(l+1)*window_size]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0 or max_ == 0.0:
                continue
            val += math.log(max_ / min_)
    return w * val

def _uism(x):
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = Rs * R
    G_edge_map = Gs * G
    B_edge_map = Bs * B
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    return 0.299*r_eme + 0.587*g_eme + 0.144*b_eme

def _uiconm(x, window_size):
    k1 = int(x.shape[1] / window_size)
    k2 = int(x.shape[0] / window_size)
    if k1 == 0 or k2 == 0:
        return 0.0
    w = -1.0 / (k1 * k2)
    x = x[:window_size*k2, :window_size*k1]
    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:(k+1)*window_size, l*window_size:(l+1)*window_size, :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if bot == 0.0 or top == 0.0:
                continue
            val += (top / bot) * math.log(top / bot)
    return w * val

def getUIQM(img_np_uint8):
    x = img_np_uint8.astype(np.float32)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 10)
    return c1*uicm + c2*uism + c3*uiconm

# ==================== 批量计算平均 UIQM ====================

def list_images(folder, exts):
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="计算文件夹中所有图像的平均 UIQM 值")
    # ✅ 在这里设置默认参数
    parser.add_argument("--dir", type=str, default="/root/autodl-tmp/sqwu/UIR/output/EUVP-ours-1-500", help="图像文件夹路径（默认: ./images）")
    parser.add_argument("--ext", type=str, nargs="+", default=[".jpg", ".png", ".jpeg"], help="文件扩展名（默认: .jpg .png .jpeg）")
    parser.add_argument("--print_each", action="store_true", help="是否逐张打印 UIQM 值（默认: False）")

    args = parser.parse_args()

    img_paths = list_images(args.dir, args.ext)
    if not img_paths:
        print(f"[警告] 文件夹 {args.dir} 下未找到图片 {args.ext}")
        return

    scores = []
    for i, path in enumerate(img_paths, 1):
        img = Image.open(path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        score = getUIQM(img_np)
        scores.append(score)
        if args.print_each:
            print(f"[{i:03d}/{len(img_paths)}] {os.path.basename(path)}: UIQM={score:.4f}")

    scores = np.array(scores)
    print("\n========== 统计结果 ==========")
    print(f"图像数量: {len(scores)}")
    print(f"平均 UIQM: {np.mean(scores):.6f}")
    print(f"标准差: {np.std(scores):.6f}")

if __name__ == "__main__":
    main()
