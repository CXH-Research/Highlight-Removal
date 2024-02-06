import numpy as np
import cv2
from skimage import morphology, color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import os
from p_tqdm import p_map

# Code for the following paper:
# H. L. Shen, H. G. Zhang, S. J. Shao, and J. H. Xin,
# Simple and Efficient Method for Specularity Removal in an Image,

threshold_chroma = 0.03
nu = 0.5
folder = 'input'

def process(image_path):
    # Read the image
    I = cv2.imread(os.path.join(folder, image_path))
    I = I.astype(np.float64)
    height, width, dim = I.shape

    # Reshape the image
    I3c = I.reshape(height*width, 3)

    # Calculate specular-free image
    Imin = np.min(I3c, axis=1)
    Imax = np.max(I3c, axis=1)
    Ithresh = np.mean(Imin) + nu * np.std(Imin)
    Iss = I3c - np.repeat(Imin[:, np.newaxis], 3, axis=1) * (Imin[:, np.newaxis] > Ithresh) + Ithresh * (Imin[:, np.newaxis] > Ithresh)

    # Calculate specular component
    IBeta = (Imin - Ithresh) * (Imin > Ithresh) + 0

    # Estimate largest region of highlight
    IHighlight = IBeta.reshape(height, width)
    # IHighlight = color.rgb2gray(IHighlight)
    IHighlight = IHighlight > threshold_otsu(IHighlight)
    IDominantRegion = morphology.remove_small_objects(IHighlight, 1, connectivity=1)

    # Dilate largest region by 5 pixels to obtain its surrounding region
    se = morphology.square(5)
    ISurroundingRegion = morphology.dilation(IDominantRegion, se)
    ISurroundingRegion = np.logical_xor(ISurroundingRegion, IDominantRegion)

    # Solve least squares problem
    Vdom = np.mean(I3c[IDominantRegion.flatten(), :], axis=0)
    Vsur = np.mean(I3c[ISurroundingRegion.flatten(), :], axis=0)
    Betadom = np.mean(IBeta[IDominantRegion.flatten()])
    Betasur = np.mean(IBeta[ISurroundingRegion.flatten()])
    k = (Vsur - Vdom) / (Betasur - Betadom)

    # Estimate diffuse and specular components
    Idf = I3c - np.min(k) * IBeta[:, np.newaxis]
    Isp = I3c - Idf

    # Display images
    # plt.figure(); plt.imshow(I.astype(np.uint8)); plt.title('Original')
    # plt.figure(); plt.imshow(Idf.reshape(height, width, 3).astype(np.uint8)); plt.title('Diffuse Component')
    # plt.figure(); plt.imshow(Isp.reshape(height, width, 3).astype(np.uint8)); plt.title('Specular Component')

    # Save images
    cv2.imwrite(os.path.join('result', image_path), Idf.reshape(height, width, 3).astype(np.uint8))
    # cv2.imwrite('comp_sp.jpg', Isp.reshape(height, width, 3).astype(np.uint8))


if __name__ == '__main__':
    imgs = os.listdir(folder)
    p_map(process, imgs)