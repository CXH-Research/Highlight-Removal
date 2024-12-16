import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import read_image, write_png
from torch.multiprocessing import Pool
from functools import partial
from sklearn.cluster import KMeans
import numpy as np

def clip(x, lb, ub):
    return torch.clamp(x, lb, ub)

def pixel_clustering(Ich_pseudo, Imask, width, height, th_chroma):
    MAX_NUM_CLUST = 100

    N = width * height

    Idone = torch.zeros(N, dtype=torch.bool, device=Ich_pseudo.device)
    Iclust = torch.zeros(N, dtype=torch.int32, device=Ich_pseudo.device)
    label = 0

    for i in range(N):
        if not Idone[i] and Imask[i]:
            c = Ich_pseudo[i]
            label += 1
            for j in range(i, N):
                if not Idone[j] and Imask[j]:
                    dist = torch.abs(c - Ich_pseudo[j]).sum()
                    if dist < th_chroma:
                        Idone[j] = True
                        Iclust[j] = label

    num_clust = label

    if num_clust > MAX_NUM_CLUST:
        return Iclust, num_clust

    clust_mean = torch.zeros((num_clust, 2), device=Ich_pseudo.device)
    num_pixel = torch.zeros(num_clust, device=Ich_pseudo.device)

    for i in range(N):
        k = Iclust[i]
        if k > 0 and k <= num_clust:
            num_pixel[k - 1] += 1
            clust_mean[k - 1] += Ich_pseudo[i]

    clust_mean = clust_mean / num_pixel[:, None]

    for i in range(N):
        if Imask[i]:
            c = Ich_pseudo[i]
            dists = torch.abs(c - clust_mean).sum(dim=1)
            label = torch.argmin(dists) + 1
            Iclust[i] = label

    return Iclust, num_clust

def Shen2013(I):
    # Ensure input is float tensor
    if not I.dtype.is_floating_point:
        I = I.float()
    assert I.min() >= 0 and I.max() <= 1, 'Input I is not within [0, 1] range.'
    n_row, n_col, n_ch = I.shape
    assert n_row > 1 and n_col > 1, 'Input I has a singleton dimension.'
    assert n_ch == 3, 'Input I is not an RGB image.'

    height, width = n_row, n_col
    I_flat = I.view(-1, 3)

    Imin, _ = torch.min(I_flat, dim=1)
    Imax, _ = torch.max(I_flat, dim=1)
    Iran = Imax - Imin

    umin_val = Imin.mean()

    Imask = Imin > umin_val

    Ich_pseudo = torch.zeros((height * width, 2), device=I.device)
    frgb = torch.zeros_like(I_flat)
    crgb = torch.zeros_like(I_flat)
    srgb = torch.zeros(height * width, device=I.device)

    frgb[Imask] = I_flat[Imask] - Imin[Imask][:, None] + umin_val
    srgb[Imask] = frgb[Imask].sum(dim=1)
    crgb[Imask] = frgb[Imask] / (srgb[Imask][:, None] + 1e-10)

    Ich_pseudo_min, _ = torch.min(crgb[Imask], dim=1)
    Ich_pseudo_max, _ = torch.max(crgb[Imask], dim=1)
    Ich_pseudo[Imask, 0] = Ich_pseudo_min
    Ich_pseudo[Imask, 1] = Ich_pseudo_max

    # Replace the clustering function with kmeans
    th_chroma = 0.3
    Iclust = torch.zeros(height * width, dtype=torch.int32, device=I.device)
    if Imask.sum() > 0:
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(Ich_pseudo[Imask].cpu().numpy())
        Iclust[Imask] = torch.from_numpy(clusters).to(I.device) + 1
    else:
        num_clust = 0

    ratio = torch.zeros(height * width, device=I.device)
    Iratio = torch.zeros(height * width, device=I.device)

    EPS = 1e-10
    th_percent = 0.5

    num_clust = Iclust.max().item()
    for k in range(1, num_clust + 1):
        idx = (Iclust == k) & (Iran > umin_val)
        if idx.sum() == 0:
            continue
        ratio_k = Imax[idx] / (Iran[idx] + EPS)
        ratio_est = torch.sort(ratio_k)[0][int(ratio_k.numel() * th_percent)]
        Iratio[Iclust == k] = ratio_est

    I_s = torch.zeros(height * width, device=I.device)
    I_d = I_flat.clone()

    idx = Imask.nonzero(as_tuple=True)[0]
    uvalue = (Imax[idx] - Iratio[idx] * Iran[idx]).clamp(min=0)
    I_s[idx] = uvalue
    fvalue = I_flat[idx] - I_s[idx][:, None]
    I_d[idx] = clip(fvalue, 0, 1)

    I_d = I_d.view(height, width, 3)

    return I_d

def fix(I, AuthorYEAR=None):
    # Ensure input is float tensor
    if not I.dtype.is_floating_point:
        I = I.float()
    if AuthorYEAR is None:
        AuthorYEAR = Shen2013

    n_row, n_col, _ = I.shape

    # DRM: I = I_d + I_s
    I_d = AuthorYEAR(I)
    I_s = clip(I - I_d, 0, 1)

    I_d_m_1 = I_d.clone()

    # Table 1 Parameters
    omega = 0.3
    k = 10
    epsilon = 0.2  # RMSE convergence criteria
    iter_count = 0
    max_iter_count = 5

    device = I.device
    H_low = torch.ones((3, 3), device=device) / 9.0
    H_h_emph = -k * H_low
    H_h_emph[1, 1] = 1 + k - k * H_low[1, 1]

    # Convolution kernels
    H_h_emph = H_h_emph.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
    Theta = clip(F.conv2d(I.permute(2, 0, 1).unsqueeze(0), H_h_emph, padding=1), 0, 1)


    while True:
        Upsilon_d = clip(F.conv2d(I_d.permute(2, 0, 1).unsqueeze(0), H_h_emph, padding=1), 0, 1)
        Upsilon_s = clip(F.conv2d(I_s.permute(2, 0, 1).unsqueeze(0), H_h_emph, padding=1), 0, 1)
        Upsilon = clip(Upsilon_d + Upsilon_s, 0, 1)

        err_diff = (Upsilon_d > Theta).sum(dim=1)[0] >= 3

        N_s = (I_s > 0).sum()
        if N_s == 0:
            break
        N_s = 2 * int((N_s.float().sqrt() / 2).ceil().item()) + 1
        center = (N_s, N_s)

        err_indices = err_diff.nonzero(as_tuple=False)

        for idx in range(err_indices.shape[0]):
            row, col = err_indices[idx]
            nh_r_start = max(0, row - center[0] // 2)
            nh_r_end = min(row + center[0] // 2 + 1, n_row)
            nh_c_start = max(0, col - center[1] // 2)
            nh_c_end = min(col + center[1] // 2 + 1, n_col)

            nh_I = I[nh_r_start:nh_r_end, nh_c_start:nh_c_end, :].reshape(-1, 3)
            nh_Theta = Theta[:, nh_r_start:nh_r_end, nh_c_start:nh_c_end].permute(1, 2, 0).reshape(-1, 3)
            nh_Upsilon = Upsilon[:, nh_r_start:nh_r_end, nh_c_start:nh_c_end].permute(1, 2, 0).reshape(-1, 3)

            center_p = I[row, col, :].unsqueeze(0)

            Phi_I = (center_p - nh_I).pow(2).sum(dim=1)
            Phi_Th_Up = (nh_Theta - nh_Upsilon).pow(2).sum(dim=1)

            Phi = omega * Phi_I + (1 - omega) * Phi_Th_Up
            plausible = Phi.argmin()
            p_row = nh_r_start + plausible // (nh_c_end - nh_c_start)
            p_col = nh_c_start + plausible % (nh_c_end - nh_c_start)

            I_d[row, col, :] = I_d_m_1[p_row, p_col, :]

        iter_count += 1

        I_s = clip(I - I_d, 0, 1)

        if torch.sqrt(F.mse_loss(I_d, I_d_m_1)) < epsilon or iter_count >= max_iter_count:
            break

        I_d_m_1 = I_d.clone()

    return I_d

def process_image(filename, Image_dir, result_dir):
    output_filename = os.path.join(result_dir, os.path.basename(filename))
    if os.path.exists(output_filename):
        print(f'Skipping {os.path.basename(filename)} (already exists in result folder)')
        return

    # Load image
    image = read_image(os.path.join(Image_dir, filename)).float() / 255.0  # Shape: [C, H, W]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert grayscale to RGB
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Permute image to [H, W, C]
    image = image.permute(1, 2, 0)

    # Apply fix function
    sfi = fix(image)

    # Permute back to [C, H, W]
    sfi = sfi.permute(2, 0, 1).cpu()

    # Save image
    write_png((sfi * 255).byte(), output_filename)

def main():
    Image_dir = './specular'
    result_dir = 'Yamamoto2019'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # List all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = [f for f in os.listdir(Image_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    # Use multiprocessing Pool to parallelize
    for image in images:
        process_image(image, Image_dir, result_dir)
    # pool = Pool()
    # func = partial(process_image, Image_dir=Image_dir, result_dir=result_dir)
    # pool.map(func, images)
    # pool.close()
    # pool.join()

if __name__ == '__main__':
    main()
