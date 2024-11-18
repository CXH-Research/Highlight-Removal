from PIL import Image
import numpy as np
import os
from p_tqdm import p_map


def getShape(img_3D):
    length = img_3D.shape[0]
    width = img_3D.shape[1]
    colorChannel = img_3D.shape[2]
    return length, width, colorChannel


def highlightDistinguish(img_3D):
    length, width, colorChannel = getShape(img_3D)
    if(colorChannel != 3):
        print("颜色通道异常！")
    else:
        img_2D = np.reshape(img_3D, (-1, 3))
        I_min = np.zeros(img_2D.shape[0])
        for i in range(I_min.shape[0]):
            I_min[i] = img_2D[i, :].min()

        T_offsetHighlight = 2*np.mean(I_min)+0.5*np.std(I_min)
        offset_highlight = np.zeros(img_2D.shape[0])
        for i in range(offset_highlight.shape[0]):
            if (I_min[i] > T_offsetHighlight):
                offset_highlight[i] = T_offsetHighlight
            else:
                offset_highlight[i] = I_min[i]

        I_highlightDetection = np.zeros(img_2D.shape[0])
        I_diffuseDetection = np.zeros(img_2D.shape[0])
        meanOfI_min = np.mean(I_min)
        for i in range(img_2D.shape[0]):
            if(offset_highlight[i] > 2*meanOfI_min):
                I_highlightDetection[i] = 1
            else:
                I_diffuseDetection[i] = 1

        T_offsetMSF = 2*np.mean(I_min)+0.5*np.std(I_min)
        offset_MSF = np.zeros(img_2D.shape[0])
        for i in range(offset_MSF.shape[0]):
            if (I_min[i] > T_offsetMSF):
                offset_MSF[i] = T_offsetMSF
            else:
                offset_MSF[i] = I_min[i]
        MSFimg_2D = img_2D.copy()
        for i in range(MSFimg_2D.shape[0]):
            MSFimg_2D[i] = MSFimg_2D[i]-I_min[i]+offset_MSF[i]
        MSFimg_3D = np.reshape(MSFimg_2D, (length, width, colorChannel))

        return I_highlightDetection, I_diffuseDetection, MSFimg_3D


def getLuminanceValue(img_clip):
    return 0.27*img_clip[0]+0.67*img_clip[1]+0.06*img_clip[2]
#     return 0.33*img_clip[0]+0.33*img_clip[1]+0.33*img_clip[2]


def getHF(MSFimg_3D, img_3D, I_highlightDetection, I_diffuseDetection):
    MSFimg_2D = np.reshape(MSFimg_3D, (-1, 3))
    img_2D = np.reshape(img_3D, (-1, 3))

    highlightImg_2D = img_2D.copy()
    diffuseImg_2D = img_2D.copy()

    for i in range(highlightImg_2D.shape[0]):
        if(I_highlightDetection[i] == 1):
            diffuseImg_2D[i] = [0, 0, 0]
            operator = 1 + \
                np.exp(-14 *
                       np.power((getLuminanceValue(highlightImg_2D[i])/255), 1.6))*1.025
            highlightImg_2D[i]*operator
        else:
            highlightImg_2D[i] = [0, 0, 0]
    HFimg_3D = np.reshape(highlightImg_2D+diffuseImg_2D, img_3D.shape)
    return HFimg_3D


def getHDR(img_path):
    img_3D = np.array(Image.open(img_path))
    I_highlightDetection, I_diffuseDetection, MSFimg = highlightDistinguish(
        img_3D)
    I_HighlightFree_3D = getHF(
        MSFimg, img_3D, I_highlightDetection, I_diffuseDetection)

    I_HighlightFree_2D = np.reshape(I_HighlightFree_3D, (-1, 3))
#     print(I_HighlightFree_2D)

    L_HighlightFree_2D = np.zeros((I_HighlightFree_2D.shape[0], 1))
    for i in range(L_HighlightFree_2D.shape[0]):
        L_HighlightFree_2D[i] = getLuminanceValue(I_HighlightFree_2D[i])
#     print(L_HighlightFree_2D)

    L_HighlightFree_sum = 0
    for i in range(I_HighlightFree_2D.shape[0]):
        L_HighlightFree_sum += np.log(0.001+L_HighlightFree_2D[i])
    L_HighlightFree_exp_logMean = np.exp(
        L_HighlightFree_sum/I_HighlightFree_2D.shape[0])
#     print(L_HighlightFree_exp_logMean)

    scaled_L_HighlightFree_2D = (
        0.07/L_HighlightFree_exp_logMean)*L_HighlightFree_2D
#     print(scaled_L_HighlightFree_2D)

    toneMapping_L_HighlightFree_2D = scaled_L_HighlightFree_2D.copy()
    for i in range(toneMapping_L_HighlightFree_2D.shape[0]):
        #         toneMapping_L_HighlightFree_2D[i] = scaled_L_HighlightFree_2D[i]/(1+scaled_L_HighlightFree_2D[i])
        toneMapping_L_HighlightFree_2D[i] = scaled_L_HighlightFree_2D[i]*(
            1+(scaled_L_HighlightFree_2D[i]/np.power(0.35, 2)))/(1+scaled_L_HighlightFree_2D[i])
#     print(toneMapping_L_HighlightFree_2D)

    HDRimg_2D = np.reshape(img_3D, (-1, 3))
    for i in range(HDRimg_2D.shape[0]):
        HDRimg_2D[i] = [0, 0, 0]
        HDRimg_2D[i] = I_HighlightFree_2D[i]*L_HighlightFree_2D[i] / \
            (255*toneMapping_L_HighlightFree_2D[i])
#         print(L_HighlightFree_2D[i]/(255*toneMapping_L_HighlightFree_2D[i]))

    HDRimg_3D = np.reshape(HDRimg_2D, img_3D.shape)
    im = Image.fromarray(HDRimg_3D)
    im.save(os.path.join(result_dir, os.path.basename(img_path)))


result_dir = 'result'

if __name__ == '__main__':
    folder = './all/SSHR'
    os.makedirs(result_dir, exist_ok=True)
    imgs = [os.path.join(folder, img) for img in os.listdir(folder)]
    p_map(getHDR, imgs, num_cpus=0.9)
