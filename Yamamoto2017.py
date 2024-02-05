import cv2
import numpy as np
import math
import os
from p_tqdm import p_map

class qx:
    TOL = 1e-4  # Example tolerance value, adjust as necessary
    SIGMAS = 3.0  # Example sigma value for space in bilateral filter, adjust as necessary
    SZ = 2*math.ceil(2*3.0)+1  # Example size for bilateral filter, adjust as necessary
    SIGMAR = 0.1  # Example sigma value for range in bilateral filter, adjust as necessary
    THR = 0.03  # Example threshold value, adjust as necessary

def qx_highlight_removal_bf(src):
    """
    Remove highlights from an image using bilateral filtering.
    
    Parameters:
    src (numpy.ndarray): Source image.
    
    Returns:
    numpy.ndarray: Image with highlights removed.
    """
    total = np.sum(src, axis=2)
    total3 = np.repeat(total[:, :, np.newaxis], 3, axis=2)

    tIdx = total <= qx.TOL
    tIdx3 = np.repeat(tIdx[:, :, np.newaxis], 3, axis=2)

    sigma = np.zeros_like(src, dtype=float)
    sigma[~tIdx3] = src[~tIdx3] / total3[~tIdx3]
    sigmaMax = np.max(sigma, axis=2)
    sigmaMin = np.min(sigma, axis=2)
    sigmaMin3 = np.repeat(sigmaMin[:, :, np.newaxis], 3, axis=2)

    sIdx = (sigmaMin >= 1/3 - qx.TOL) & (sigmaMin <= 1/3 + qx.TOL)
    sIdx3 = np.repeat(sIdx[:, :, np.newaxis], 3, axis=2)

    lambda_val = np.ones_like(src, dtype=float) / 3
    lambda_val[~sIdx3] = (sigma[~sIdx3] - sigmaMin3[~sIdx3]) / (3 * (lambda_val[~sIdx3] - sigmaMin3[~sIdx3]))
    lambdaMax = np.max(lambda_val, axis=2)

    while True:
        sigmaMaxF = cv2.bilateralFilter(sigmaMax.astype(np.float32), qx.SZ, qx.SIGMAS, qx.SIGMAR)
        if np.count_nonzero(sigmaMaxF - sigmaMax > qx.THR) == 0:
            break
        sigmaMax = np.maximum(sigmaMax, sigmaMaxF)

    zIdx = (sigmaMax >= 1/3 - qx.TOL) & (sigmaMax <= 1/3 + qx.TOL)

    srcMax = np.max(src, axis=2)

    sfi = np.zeros(src.shape[:2], dtype=float)
    sfi[~zIdx] = (srcMax[~zIdx] - sigmaMax[~zIdx] * total[~zIdx]) / (1 - 3 * sigmaMax[~zIdx])
    sfi3 = np.repeat(sfi[:, :, np.newaxis], 3, axis=2)

    mIdx = sigmaMax <= 1/3 + qx.TOL
    mIdx3 = np.repeat(mIdx[:, :, np.newaxis], 3, axis=2)

    dst = src.copy()
    dst[~mIdx3] = src[~mIdx3] - sfi3[~mIdx3]

    return dst

def Saturate(X):
    """
    Saturate the input value(s) between 0 and 1.

    Parameters:
    X : float or numpy.ndarray
        The input value or array of values to be saturated.

    Returns:
    Y : float or numpy.ndarray
        The saturated value(s), constrained between 0 and 1.
    """
    Y = np.minimum(1, np.maximum(0, X))
    return Y

folder = 'input'

def process(img_path):
    # Import image to workspace
    i_input = cv2.imread(os.path.join(folder, img_path))
    i_input = cv2.normalize(i_input.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    nRow, nCol, nCh = i_input.shape

    # i = i_d + i_s
    i_d = qx_highlight_removal_bf(i_input)
    i_s = np.minimum(1, np.maximum(0, (i_input - i_d)))

    # Iteration constraints
    iterCount = 0
    maxIterCount = 10
    epsilon = 0.2

    # While loop
    while True:
        k = 10
        h = np.ones((3, 3)) / 9

        i_input_bf = cv2.filter2D(i_input, -1, h, borderType=cv2.BORDER_REFLECT)
        i_d_bf = cv2.filter2D(i_d, -1, h, borderType=cv2.BORDER_REFLECT)
        i_s_bf = cv2.filter2D(i_s, -1, h, borderType=cv2.BORDER_REFLECT)

        i_input_um = Saturate(i_input + k * Saturate(i_input - i_input_bf))
        i_d_um = Saturate(i_d + k * Saturate(i_d - i_d_bf))
        i_s_um = Saturate(i_s + k * Saturate(i_s - i_s_bf))

        i_combined_um = Saturate(i_d_um + i_s_um)

        aux = i_s
        omega = 0.3

        replaceThese = np.all(i_d_um > i_input_um, axis=2)

        row, col = np.where(replaceThese)

        paux = aux

        for ind in range(np.count_nonzero(replaceThese)):
            if row[ind] == 0 or col[ind] == 0 or row[ind] == nRow - 1 or col[ind] == nCol - 1:
                continue

            Y_i_input = i_input[row[ind]-1:row[ind]+2, col[ind]-1:col[ind]+2, :]
            Y_i_input_um = i_input_um[row[ind]-1:row[ind]+2, col[ind]-1:col[ind]+2, :]
            Y_i_combined_um = i_combined_um[row[ind]-1:row[ind]+2, col[ind]-1:col[ind]+2, :]

            Y_i_input_col = Y_i_input.reshape(9, 3)
            Y_i_input_um_col = Y_i_input_um.reshape(9, 3)
            Y_i_combined_um_col = Y_i_combined_um.reshape(9, 3)

            E_input_col = np.sum((Y_i_input_col[4] - Y_i_input_col)**2, axis=1)
            E_um_col = np.sum((Y_i_input_um_col - Y_i_combined_um_col)**2, axis=1)

            E_pp_col = omega * E_input_col + (1 - omega) * E_um_col

            plausible = np.argmin(E_pp_col)
            pRow, pCol = np.unravel_index(plausible, (3, 3))

            aux[row[ind], col[ind]] = paux[row[ind]-1+pRow, col[ind]-1+pCol]

        if np.sqrt(np.mean((aux - paux)**2)) < epsilon or iterCount >= maxIterCount:
            break

        i_s = aux
        i_d = Saturate(i_input - i_s)
        cv2.imwrite(os.path.join('result', img_path), i_d)

if __name__ == '__main__':
    imgs = os.listdir(folder)
    p_map(process, imgs, num_cpus=0.9)