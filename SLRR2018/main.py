# code=utf-8
"""
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24
# @Author  : Guanglei Ding
# @Site    : http://openaccess.thecvf.com/content_ECCV_2018/papers/Jie_Guo_Single_Image_Highlight_ECCV_2018_paper.pdf
# @File    : main.py
# @description: This is a NON-OFFICIAL python implementation of 'Single Image Highlight Removal with a Sparse and Low-Rank Reflection Model'
"""
# import numpy as cp
import cupy as cp
import cv2
from SLRR_pytorch import SLRR
from extract_colors import extract_colors
import psutil
import os
from tqdm import tqdm

K = 50
iteration = 250


Gamma = cp.ones((3, 1), dtype=cp.float32) * 1 / 3  # (1 ,3)

folder = 'test_imgs'
imgs = os.listdir(folder)

img_save_dir = 'result'
os.makedirs(img_save_dir, exist_ok=True)

for img_read_path in tqdm(imgs):
    filename = os.path.basename(img_read_path)

    img = cv2.imread(os.path.join(folder, img_read_path))  # H, W, C
    #H, W, C = (128, 128, 3)
    H, W, C = img.shape  # 360,260,3 requires 32GB mem!!!
    # img = cv2.resize(img, (W, H))


    color_dics = extract_colors(cp.asarray(img), K)  # (3,K)
    X = img.astype(cp.float32).reshape((-1, 3)).T / 255.0  # N, 3 ->  3, N
    # print(X.shape, color_dics.shape)  # (3,N) ,  (3,K)

    Phi_d, Wd, Ms = SLRR(X, color_dics, Gamma=Gamma, iteration=iteration)
    Ms = cp.asarray(Ms)
    Hlt = cp.dot(Gamma, Ms).T
    Hlt = Hlt.reshape((H, W, 3))
    Hlt = cp.clip(Hlt * 255, a_min=0, a_max=255).astype(cp.uint8)
    # cv2.imwrite(os.path.join(img_save_dir, "Hlt.png"), Hlt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0.3])
    Hlt = Hlt.get()
    diffuse = cv2.subtract(img, Hlt)
    cv2.imwrite(os.path.join(img_save_dir, filename), diffuse)