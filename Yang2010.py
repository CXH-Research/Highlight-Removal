import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from scipy.interpolate import interpn
from scipy.ndimage import convolve
from scipy.signal import gaussian
from PIL import Image
from p_tqdm import p_map


# def bilateralFilter(data, edge=None, edgeMin=None, edgeMax=None, sigmaSpatial=None, sigmaRange=None,
#                     samplingSpatial=None, samplingRange=None):
#     """
#     Applies a bilateral filter to the input data.

#     Parameters:
#     - data: 2D numpy array representing the grayscale image.
#     - edge: 2D numpy array representing the edge image. If None, uses data as the edge image.
#     - edgeMin: Minimum value of the edge image. If None, calculates from the edge image.
#     - edgeMax: Maximum value of the edge image. If None, calculates from the edge image.
#     - sigmaSpatial: Spatial standard deviation. If None, calculates based on the image size.
#     - sigmaRange: Range standard deviation. If None, calculates based on edgeMin and edgeMax.
#     - samplingSpatial: Spatial sampling rate. If None, equals sigmaSpatial.
#     - samplingRange: Range sampling rate. If None, equals sigmaRange.

#     Returns:
#     - output: 2D numpy array after applying the bilateral filter.
#     """

#     if data.ndim > 2:
#         raise ValueError('data must be a greyscale image with size [height, width]')

#     if not data.dtype == np.float64:
#         raise ValueError('data must be of class "double"')

#     if edge is None:
#         edge = data

#     if edge.ndim > 2:
#         raise ValueError('edge must be a greyscale image with size [height, width]')

#     if not edge.dtype == np.float64:
#         raise ValueError('edge must be of class "double"')

#     inputHeight, inputWidth = data.shape

#     if edgeMin is None:
#         edgeMin = np.min(edge)
#         print(f'edgeMin not set! Defaulting to: {edgeMin}')

#     if edgeMax is None:
#         edgeMax = np.max(edge)
#         print(f'edgeMax not set! Defaulting to: {edgeMax}')

#     edgeDelta = edgeMax - edgeMin

#     if sigmaSpatial is None:
#         sigmaSpatial = min(inputWidth, inputHeight) / 16
#         print(f'Using default sigmaSpatial of: {sigmaSpatial}')

#     if sigmaRange is None:
#         sigmaRange = 0.1 * edgeDelta
#         print(f'Using default sigmaRange of: {sigmaRange}')

#     if samplingSpatial is None:
#         samplingSpatial = sigmaSpatial

#     if samplingRange is None:
#         samplingRange = sigmaRange

#     if data.shape != edge.shape:
#         raise ValueError('data and edge must be of the same size')

#     # Parameters
#     derivedSigmaSpatial = sigmaSpatial / samplingSpatial
#     derivedSigmaRange = sigmaRange / samplingRange

#     paddingXY = int(np.floor(2 * derivedSigmaSpatial)) + 1
#     paddingZ = int(np.floor(2 * derivedSigmaRange)) + 1

#     # Allocate 3D grid
#     downsampledWidth = int(np.floor((inputWidth - 1) / samplingSpatial)) + 1 + 2 * paddingXY
#     downsampledHeight = int(np.floor((inputHeight - 1) / samplingSpatial)) + 1 + 2 * paddingXY
#     downsampledDepth = int(np.floor(edgeDelta / samplingRange)) + 1 + 2 * paddingZ

#     gridData = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))
#     gridWeights = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))

#     # Compute downsampled indices
#     jj, ii = np.meshgrid(np.arange(inputWidth), np.arange(inputHeight))

#     di = np.round(ii / samplingSpatial).astype(int) + paddingXY + 1
#     dj = np.round(jj / samplingSpatial).astype(int) + paddingXY + 1
#     dz = np.round((edge - edgeMin) / samplingRange).astype(int) + paddingZ + 1

#     # Perform scatter
#     for k in range(len(dz.ravel())):
#         dataZ = data.ravel()[k]
#         if not np.isnan(dataZ):
#             dik = di.ravel()[k]
#             djk = dj.ravel()[k]
#             dzk = dz.ravel()[k]

#             gridData[dik, djk, dzk] += dataZ
#             gridWeights[dik, djk, dzk] += 1

#     # Make Gaussian kernel
#     kernelWidth = int(2 * derivedSigmaSpatial + 1)
#     kernelHeight = kernelWidth
#     kernelDepth = int(2 * derivedSigmaRange + 1)

#     halfKernelWidth = kernelWidth // 2
#     halfKernelHeight = kernelHeight // 2
#     halfKernelDepth = kernelDepth // 2

#     gridX, gridY, gridZ = np.meshgrid(np.arange(kernelWidth), np.arange(kernelHeight), np.arange(kernelDepth), indexing='ij')
#     gridX = gridX - halfKernelWidth
#     gridY = gridY - halfKernelHeight
#     gridZ = gridZ - halfKernelDepth
#     gridRSquared = (gridX**2 + gridY**2) / (derivedSigmaSpatial**2) + (gridZ**2) / (derivedSigmaRange**2)
#     kernel = np.exp(-0.5 * gridRSquared)

#     # Convolve
#     blurredGridData = convolve(gridData, kernel, mode='constant', cval=0)
#     blurredGridWeights = convolve(gridWeights, kernel, mode='constant', cval=0)

#     # Divide
#     blurredGridWeights[blurredGridWeights == 0] = -2  # Avoid divide by 0, won't read there anyway
#     normalizedBlurredGrid = blurredGridData / blurredGridWeights
#     normalizedBlurredGrid[blurredGridWeights < -1] = 0  # Put 0s where it's undefined

#     # Upsample
#     jj, ii = np.meshgrid(np.arange(inputWidth), np.arange(inputHeight), indexing='ij')
#     di = (ii / samplingSpatial) + paddingXY + 1
#     dj = (jj / samplingSpatial) + paddingXY + 1
#     dz = (edge - edgeMin) / samplingRange + paddingZ + 1

#     # Interpolate
#     points = (np.arange(normalizedBlurredGrid.shape[0]), np.arange(normalizedBlurredGrid.shape[1]), np.arange(normalizedBlurredGrid.shape[2]))
#     output = interpn(points, normalizedBlurredGrid, np.stack((di, dj, dz), axis=-1), method='linear', bounds_error=False, fill_value=0)

#     return output

folder = 'input'

# def Yang2010(image_path):
#     I = Image.open(os.path.join(folder, image_path))
#     I = np.array(I) / 255
#     """
#     Yang2010 I_d = Yang2010(I)
#     This method uses a fast bilateralFilter implementation.
    
#     This method should have equivalent functionality as
#     `qx_highlight_removal_bf.cpp` formerly distributed by the author.
    
#     See also SIHR, Tan2005.
#     """
#     # assert I.dtype == np.float, 'Input I is not type double.'
#     assert np.min(I) >= 0 and np.max(I) <= 1, 'Input I is not within [0, 1] range.'
#     n_row, n_col, n_ch = I.shape
#     assert n_row > 1 and n_col > 1, 'Input I has a singleton dimension.'
#     assert n_ch == 3, 'Input I is not a RGB image.'

#     total = np.sum(I, axis=2)

#     sigma = I / total[:,:,None]
#     sigma[np.isnan(sigma)] = 0

#     sigmaMin = np.min(sigma, axis=2)
#     sigmaMax = np.max(sigma, axis=2)

#     lambda_ = np.ones_like(I) / 3
#     lambda_ = (sigma - sigmaMin[:,:,None]) / (3 * (lambda_ - sigmaMin[:,:,None]))
#     lambda_[np.isnan(lambda_)] = 1 / 3

#     lambdaMax = np.max(lambda_, axis=2)

#     SIGMAS = 0.25 * min(n_row, n_col)
#     SIGMAR = 0.04
#     THR = 0.03

#     while True:
#         sigmaMaxF = bilateralFilter(sigmaMax, lambdaMax, 0, 1, SIGMAS, SIGMAR)
#         if np.count_nonzero(sigmaMaxF-sigmaMax > THR) == 0:
#             break
#         sigmaMax = np.maximum(sigmaMax, sigmaMaxF)

#     Imax = np.max(I, axis=2)

#     den = (1 - 3 * sigmaMax)
#     I_s = (Imax - sigmaMax * total) / den
#     I_s[den == 0] = np.max(I_s[den != 0])

#     I_d = np.minimum(1, np.maximum(0, I-I_s[:,:,None]))
    
#     I_d = Image.fromarray((I_d * 255).astype(np.uint8))
#     I_d.save(os.path.join('result', image_path))
#     return I_d

import numpy as np
import cv2

def Yang2010(image_path):
    """
    Yang2010 I_d = Yang2010(I)
    This method uses a fast bilateralFilter implementation.
    It should have equivalent functionality as `qx_highlight_removal_bf.cpp` formerly distributed by the author.
    
    Parameters:
    I: Input image as a numpy array of type double, with values in [0, 1] range and shape (n_row, n_col, 3).
    
    Returns:
    I_d: The processed image with the same shape as input.
    """
    I = Image.open(os.path.join(folder, image_path))
    I = np.array(I) / 255
    assert I.dtype == np.float64, 'Input I is not type double.'
    assert I.min() >= 0 and I.max() <= 1, 'Input I is not within [0, 1] range.'
    n_row, n_col, n_ch = I.shape
    assert n_row > 1 and n_col > 1, 'Input I has a singleton dimension.'
    assert n_ch == 3, 'Input I is not a RGB image.'

    total = np.sum(I, axis=2)

    sigma = I / total[:,:,None]
    sigma[np.isnan(sigma)] = 0

    sigmaMin = np.min(sigma, axis=2)
    sigmaMax = np.max(sigma, axis=2)

    lambda_ = np.ones(I.shape) / 3
    lambda_ = (sigma - sigmaMin[:,:,None]) / (3 * (lambda_ - sigmaMin[:,:,None]))
    lambda_[np.isnan(lambda_)] = 1 / 3

    lambdaMax = np.max(lambda_, axis=2)

    SIGMAS = 0.25 * min(I.shape[0], I.shape[1])
    SIGMAR = 0.04
    THR = 0.03

    while True:
        sigmaMaxF = cv2.bilateralFilter(sigmaMax.astype(np.float32), -1, SIGMAR, SIGMAS)
        if np.count_nonzero(sigmaMaxF - sigmaMax > THR) == 0:
            break
        sigmaMax = np.maximum(sigmaMax, sigmaMaxF)

    Imax = np.max(I, axis=2)

    den = (1 - 3 * sigmaMax)
    I_s = (Imax - sigmaMax * total) / den
    I_s[den == 0] = np.max(I_s[den != 0])

    I_d = np.minimum(1, np.maximum(0, I - I_s[:,:,None]))

    I_d = Image.fromarray((I_d * 255).astype(np.uint8))
    
    I_d.save(os.path.join('result', image_path))

    return I_d


if __name__ == '__main__':
    imgs = os.listdir(folder)
    p_map(Yang2010, imgs, num_cpus=0.9)