# Code for the following paper:
# H. L. Shen, H. G. Zhang, S. J. Shao, and J. H. Xin,
# Chromaticity-based separation of reflection components in a single image,
# Pattern Recognition, 41(8), 2461-2469, 2008.
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from p_tqdm import p_map

threshold_chroma = 0.03

folder = 'input'

def process(image_path):
    # Load image
    I = Image.open(os.path.join(folder, image_path))
    I = np.array(I, dtype=np.float32)
    height, width, dim = I.shape

    I3c = I.reshape(height*width, 3)

    # Calculate specular-free image
    Imin = np.min(I3c, axis=1)
    Imax = np.max(I3c, axis=1)
    Iss = I3c - np.tile(Imin[:, np.newaxis], (1, 3)) + np.mean(Imin)
    Itemp = I3c - Iss

    # Calculate the mask of combined pixels and diffuse pixels
    th = np.mean(Imin)
    Imask_cmb = np.zeros(height * width)
    ind_cmb = np.where((Itemp[:,0] > th) & (Itemp[:,1] > th) & (Itemp[:,2] > th))[0]
    Imask_cmb[ind_cmb] = 1

    Imask_df = np.zeros(height * width)
    ind_df = np.where((Itemp[:,0] < th) & (Itemp[:,1] < th) & (Itemp[:,2] < th) & (Imax > 20))[0]
    Imask_df[ind_df] = 1

    # Calculate chromaticity
    Ichroma = Iss / np.tile(np.sum(Iss, axis=1)[:, np.newaxis], (1, 3))

    # Specularity removal
    # Find the pixels that need processing
    Imask_all = np.zeros(height * width)  # Pixels with combined reflection or diffuse reflection
    Imask_processed = -1 * np.ones(height * width)  # Processed pixel
    ind_all = np.where((Imask_cmb == 1) | (Imask_df == 1))[0]
    Imask_all[ind_all] = 1
    Imask_processed[ind_all] = 0  # -1: not considered; 0: not processed; 1: processed

    vs = np.array([255, 255, 255])  # Light color, assumed white

    Idf = I3c.copy()
    Isp = np.zeros_like(I3c)

    Icoef = np.zeros((height * width, 2))  # The diffuse and specular coefficient of each pixel

    while True:
        # If all pixels are processed, break
        ind = np.where(Imask_processed == 0)[0]
        if len(ind) == 0:
            break
        
        # Find the diffuse pixels from the un-processed ones
        ind_0 = np.where((Imask_processed == 0) & (Imask_df == 1))[0]
        
        if len(ind_0) > 0:  # If there are un-processed diffuse pixels
            # Find the pixel with maximum rgb values
            Imax_sub = Imax[ind_0]
            Y = np.max(Imax_sub)
            ind = np.where((Imax == Y) & (Imask_processed == 0) & (Imask_df == 1))[0]
            ind = ind[0]
            
            # Regard the pixel as body color
            vb = Idf[ind]
            cb = Ichroma[ind]
            vcomb = np.vstack([vb, vs]).T

            # Chromaticity difference
            c_diff = Ichroma - np.tile(cb, (height * width, 1))
            c_diff_sum = np.sum(np.abs(c_diff), axis=1)
            
            # Exclude non-diffuse and non-combined reflection pixels
            ind = np.arange(height * width)
            ind = np.delete(ind, ind_all)
            c_diff_sum[ind] = 999
            
            # Find pixels with chromaticity difference < threshold_chroma
            ind = np.where((c_diff_sum < threshold_chroma) & (Imask_processed == 0))[0]
            
            # Let the pixel be the diffuse component, then solve the reflection coefficient
            if len(ind) > 0:
                v = I3c[ind].T
                coef = np.linalg.pinv(vcomb) @ v
                ind_c = np.where(coef[1] < 0)[0]
                if len(ind_c) > 0:
                    v_c = v[:, ind_c]
                    coef_c = np.linalg.pinv(vb[:, np.newaxis]) @ v_c
                    coef[:, ind_c] = np.vstack([coef_c, np.zeros(len(ind_c))])
                
                coef[1] = 0
                v_df = vcomb @ coef
                
                Icoef[ind] = coef.T
                Idf[ind] = v_df.T  # Diffuse component
                Isp[ind] = I3c[ind] - Idf[ind]  # Specular component
                Imask_processed[ind] = 1
                
                # Display how many pixels are processed
                ind = np.where(Imask_processed == 1)[0]
                # print(f'{len(ind)} / {len(ind_all)}')
                
        else:  # If all diffuse pixel have been processed
            # Find combined pixels
            ind = np.where((Imask_processed == 0) & (Imask_cmb == 1))[0]
            if len(ind) == 0:
                break
            
            # Calculate chromaticity difference
            ind = ind[0]
            cb = Ichroma[ind]
            
            c_diff = Ichroma - np.tile(cb, (height * width, 1))
            c_diff_sum = np.sum(np.abs(c_diff), axis=1)
            
            # Find diffuse pixel with closest chromaticity
            ind = np.arange(height * width)
            ind = np.delete(ind, ind_df)
            c_diff_sum[ind] = 999
            
            Y, ind = np.min(c_diff_sum), np.argmin(c_diff_sum)
            
            if ind > 0:
                vb = Idf[ind]
                cb = Ichroma[ind]
                vcomb = np.vstack([vb, vs]).T
                
                c_diff = Ichroma - np.tile(cb, (height * width, 1))
                c_diff_sum = np.sum(np.abs(c_diff), axis=1)
                
                # Get unprocessed pixel with similar chromaticity
                ind = np.where((Imask_processed == 0) & (c_diff_sum < Y + 0.1 * threshold_chroma))[0]
                
                if len(ind) > 0:
                    v = I3c[ind].T
                    coef = np.linalg.pinv(vcomb) @ v
                    
                    coef[1] = 0
                    v_df = vcomb @ coef
                    
                    Icoef[ind] = coef.T
                    Idf[ind] = v_df.T
                    Isp[ind] = I3c[ind] - Idf[ind]
                    Imask_processed[ind] = 1
                    
                    # Display processed pixel number
                    ind = np.where(Imask_processed == 1)[0]
                    # print(f'{len(ind)} / {len(ind_all)}')
    Image.fromarray(np.uint8(Idf.reshape(height, width, 3))).save(os.path.join('result', image_path))
    # Display images
    # plt.figure(); plt.imshow(np.uint8(I3c.reshape(height, width, 3))); plt.title('Original')
    # plt.figure(); plt.imshow(np.uint8(Idf.reshape(height, width, 3))); plt.title('Diffuse Component')
    # plt.figure(); plt.imshow(np.uint8(Isp.reshape(height, width, 3))); plt.title('Specular Component')
    # plt.show()

    # Optionally save images
    # Image.fromarray(np.uint8(Idf.reshape(height, width, 3))).save('comp_df.bmp')
    # Image.fromarray(np.uint8(Isp.reshape(height, width, 3))).save('comp_sp.bmp')


if __name__ == '__main__':
    imgs = os.listdir(folder)
    p_map(process, imgs)