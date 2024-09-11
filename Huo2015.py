from PIL import Image
import numpy as np
import os
from p_tqdm import p_map

folder = './data'

def getGray(img):
    return img[:,:,0]*0.33+img[:,:,1]*0.33+img[:,:,2]*0.33

def highlightDistinguish(img_path):
    img = np.array(Image.open(os.path.join(folder, img_path)))
    N = img.shape[0] * img.shape[1]
    SF = img.copy()
    MSF = img.copy()
    plainArea = img.copy()
    highlightArea = img.copy()
    temp_sum = 0
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            temp_sum = temp_sum+img[x,y,:].min()
            
    I_min = np.array([temp_sum/N,temp_sum/N,temp_sum/N])
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixel_min = np.array([img[x,y,:].min(),img[x,y,:].min(),img[x,y,:].min()])
            SF[x,y,:] = img[x,y,:] - pixel_min
            MSF[x,y,:] = SF[x,y,:] + I_min

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(((img[x,y,:]-I_min)>I_min).all()):
                plainArea[x,y,:]=[0,0,0]
            else:
                highlightArea[x,y,:]=[0,0,0]
    im = Image.fromarray(plainArea)
    im.save(os.path.join('result', img_path))
    return highlightArea, plainArea, MSF

def getHDR(img):
    N = img.shape[0]*img.shape[1]
    L_highlight, L_plain, MSF = highlightDistinguish(img)
    sum_r = 0
    sum_g = 0
    sum_b = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            sum_r = sum_r+(np.log(MSF[x,y,0].astype(np.float32)+1/256))/N
#             print(sum_r)
            sum_g = sum_g+(np.log(MSF[x,y,1].astype(np.float32)+1/256))/N
            sum_b = sum_b+(np.log(MSF[x,y,2].astype(np.float32)+1/256))/N
    average_luminance_value = np.array([np.exp(sum_r),np.exp(sum_g),np.exp(sum_b)])
    
    return average_luminance_value

if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    imgs = os.listdir(folder)
    p_map(highlightDistinguish, imgs, num_cpus=0.9)