import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def fix(fname):
    """
    Process the image to remove specular reflections.
    
    Parameters:
    fname (str): File path of the image to process.
    
    Returns:
    numpy.ndarray: Processed image array with specular reflections removed.
    """
    I = np.asarray(Image.open(fname)).astype(np.float64) / 255.0
    # Create specular-free two-band image
    Isf = I - np.min(I, axis=2, keepdims=True)
    # Get its chroma
    den = np.tile(np.sum(Isf, axis=2, keepdims=True), (1, 1, 3))
    zero = den == 0
    c = np.zeros_like(Isf)
    c[~zero] = Isf[~zero] / den[~zero]
    cr = c[:, :, 0]
    cg = c[:, :, 1]
    # Dimensions
    nRow, nCol, nCh = I.shape
    # Reshape to column vector (easier indexing)
    I_col = I.reshape((-1, nCh))
    Isf_col = Isf.reshape((-1, nCh))
    cr_col = cr.flatten()
    cg_col = cg.flatten()
    skip = np.zeros(nRow * nCol, dtype=bool)
    # Chroma threshold values (color discontinuity)
    thR, thG = 0.05, 0.05
    # Iterates until only diffuse pixels are left
    count = 0
    iter = 0
    while True:
        for x1 in range(nRow * nCol - 1):
            x2 = x1 + 1
            if skip[x1]:
                continue
            elif np.sum(Isf_col[x2]) == 0 or np.sum(I_col[x2]) == 0 or \
                    (abs(cr_col[x1] - cr_col[x2]) > thR and abs(cg_col[x1] - cg_col[x2]) > thG):
                skip[x1] = True
                continue
            # Get local r_{d+s} ratio and r_{d}
            rd = np.sum(Isf_col[x1]) / np.sum(Isf_col[x2])
            rds = np.sum(I_col[x1]) / np.sum(I_col[x2])
            # Compare ratios and decrease intensity
            if rds > rd:
                m = np.sum(I_col[x1, 1]) - rd * np.sum(I_col[x2, 1])
                if m < 1e-3:
                    continue
                I_col[x1, :] = np.maximum(0, I_col[x1, :] - m / 3)
                count += 1
            elif rds < rd:
                m = np.sum(I_col[x2, 1]) - np.sum(I_col[x1, 1]) / rd
                if m < 1e-3:
                    continue
                I_col[x2, :] = np.maximum(0, I_col[x2, :] - m / 3)
                count += 1
        if count == 0 or iter == 1000:
            break
        count = 0
        iter += 1
    # Return diffuse image
    Idiff = I_col.reshape((nRow, nCol, nCh))
    return Idiff

# Main script
Image_dir = 'input'
result_dir = os.path.join(Image_dir, 'result')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for filename in os.listdir(Image_dir):
    if filename.upper().endswith('G'):  # Assuming 'G' is part of the file extension check
        full_path = os.path.join(Image_dir, filename)
        processed_image = fix(full_path)
        # Save the processed image
        Image.fromarray((processed_image * 255).astype(np.uint8)).save(os.path.join(result_dir, filename))
