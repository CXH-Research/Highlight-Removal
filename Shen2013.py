import numpy as np
from PIL import Image
import os
from p_tqdm import p_map

folder = 'input'

def Shen2013(image_path):
    """
    Shen2013 I_d = Shen2013(I)
    You can optionally edit the code to use kmeans instead of the clustering
    function proposed by the author.
    
    This method should have equivalent functionality as
    `sp_removal.cpp` distributed by the author.
    
    See also SIHR, Shen2008, Shen2009.
    """
    I = Image.open(os.path.join(folder, image_path))
    I = np.array(I) / 255
    # assert isinstance(I, (float, np.float32, np.float64)), 'Input I is not type single nor double.'
    assert np.min(I) >= 0 and np.max(I) <= 1, 'Input I is not within [0, 1] range.'
    
    n_row, n_col, n_ch = I.shape
    assert n_row > 1 and n_col > 1, 'Input I has a singleton dimension.'
    assert n_ch == 3, 'Input I is not a RGB image.'
    
    height, width, _ = I.shape
    I = np.reshape(I, (height * width, 3))
    
    Imin = np.min(I, axis=1)
    Imax = np.max(I, axis=1)
    Iran = Imax - Imin
    
    umin_val = np.mean(Imin)
    
    Imask = Imin > umin_val
    
    Ich_pseudo = np.zeros((height * width, 2))
    frgb = np.zeros((height * width, 3))
    crgb = frgb.copy()
    srgb = np.zeros((height * width, 1))
    
    Imin = np.expand_dims(Imin, axis=1)
    frgb[Imask, :] = I[Imask, :] - Imin[Imask] + umin_val
    srgb[Imask] = np.expand_dims(np.sum(frgb[Imask, :], axis=1), axis=1)
    crgb[Imask, :] = frgb[Imask, :] / srgb[Imask]
    
    Ich_pseudo[Imask, 0] = np.minimum(np.minimum(crgb[Imask, 0], crgb[Imask, 1]), crgb[Imask, 2])
    Ich_pseudo[Imask, 1] = np.maximum(np.maximum(crgb[Imask, 0], crgb[Imask, 1]), crgb[Imask, 2])
    
    th_chroma = 0.3
    Iclust, num_clust = pixel_clustering(Ich_pseudo, Imask, width, height, th_chroma)
    
    ratio = np.zeros((height * width, 1))
    Iratio = np.zeros((height * width, 1))
    
    N = width * height
    EPS = 1e-10
    th_percent = 0.5
    
    for k in range(1, num_clust+1):
        num = 0
        for i in range(N):
            if Iclust[i] == k and Iran[i] > umin_val:
                ratio[num] = Imax[i] / (Iran[i] + EPS)
                num += 1
        
        if num == 0:
            continue
        
        tmp = np.sort(ratio[:num])
        ratio_est = tmp[int(num * th_percent)]
        
        for i in range(N):
            if Iclust[i] == k:
                Iratio[i] = ratio_est
    
    I_s = np.zeros((height * width, 1))
    I_d = I.copy()
    
    for i in range(N):
        if Imask[i] == 1:
            uvalue = (Imax[i] - Iratio[i] * Iran[i])
            I_s[i] = max(uvalue, 0)
            fvalue = I[i, 0] - I_s[i]
            I_d[i, 0] = clip(fvalue, 0, 1)
            fvalue = I[i, 1] - I_s[i]
            I_d[i, 1] = clip(fvalue, 0, 1)
            fvalue = I[i, 2] - I_s[i]
            I_d[i, 2] = clip(fvalue, 0, 1)
    
    I_d = np.reshape(I_d, (height, width, 3))
    
    I_d = Image.fromarray((I_d * 255).astype(np.uint8))
    
    I_d.save(os.path.join('result', image_path))

    return I_d

def pixel_clustering(Ich_pseudo, Imask, width, height, th_chroma):
    MAX_NUM_CLUST = 100
    
    label = 0
    c = np.zeros(2)
    
    clust_mean = np.zeros((MAX_NUM_CLUST, 2))
    num_pixel = np.zeros(MAX_NUM_CLUST)
    
    N = width * height
    
    Idone = np.zeros(height * width, dtype=bool)
    Iclust = np.zeros(height * width, dtype=np.uint8)
    
    for i in range(N):
        if not Idone[i] and Imask[i]:
            c[0] = Ich_pseudo[i, 0]
            c[1] = Ich_pseudo[i, 1]
            label += 1
            for j in range(i, N):
                if not Idone[j] and Imask[j]:
                    dist = abs(c[0] - Ich_pseudo[j, 0]) + abs(c[1] - Ich_pseudo[j, 1])
                    if dist < th_chroma:
                        Idone[j] = True
                        Iclust[j] = label
    
    num_clust = label
    
    if num_clust > MAX_NUM_CLUST:
        return Iclust, num_clust
    
    for i in range(N):
        k = Iclust[i]
        if 1 <= k <= num_clust:
            num_pixel[k] += 1
            clust_mean[k, 0] += Ich_pseudo[i, 0]
            clust_mean[k, 1] += Ich_pseudo[i, 1]
    
    for k in range(1, num_clust+1):
        clust_mean[k, 0] /= num_pixel[k]
        clust_mean[k, 1] /= num_pixel[k]
    
    for i in range(N):
        if Imask[i]:
            c[0] = Ich_pseudo[i, 0]
            c[1] = Ich_pseudo[i, 1]
            dist_min = abs(c[0] - clust_mean[2, 0]) + abs(c[1] - clust_mean[2, 1])
            label = 1
            for k in range(2, num_clust+1):
                dist = abs(c[0] - clust_mean[k, 0]) + abs(c[1] - clust_mean[k, 1])
                if dist < dist_min:
                    dist_min = dist
                    label = k
            Iclust[i] = label
    
    return Iclust, num_clust

def clip(x, lb, ub):
    return min(ub, max(lb, x))


if __name__ == '__main__':
    imgs = os.listdir(folder)
    p_map(Shen2013, imgs, num_cpus=0.9)