import numpy as np
import cv2
import torch
import os


def get_flops(n, r, A, B):
    a = r * 3 * n + r * 3 * r + r * r * n
    b = a + 3 * 1 * n + (r - 1) * n * (r - 1) + 3 * n * (r - 1) + 3 * 3 * (r - 1) + 3 * (r - 1) * (r - 1) + 3 * (
            r - 1) * (r - 1) + 3 * 3 * (r - 1)
    s = A * a + B * b
    return '{:.5E}'.format(s)


# NMF = Non-negative Matrix Factorization
def nmf(img, illum, lamda, r, iter_WH, iter_H, use_gpu=False):
    h, w = img.shape[:2]

    # get V
    # mask = np.ones_like(img)
    # i_, j_ = np.where(mask >= 0)
    n = img.size // 3
    # mask_index = i_*w + j_
    V = img.astype('float32').reshape((-1, 3)).T

    # init W and H
    np.random.seed(100)

    W = np.random.random((3, r)).astype('float32') * 10
    W[:, 0] = illum
    W = W / (np.sum(W ** 2, axis=0) ** 0.5 + 1e-9)

    H = np.random.random_sample((r, n)).astype('float32') * 10
    ones = np.ones((3, 3), dtype='float32')

    if use_gpu:
        dev = torch.device('cuda')
        torch.cuda.empty_cache()
        H = torch.from_numpy(H).float().to(dev)
        W = torch.from_numpy(W).float().to(dev)
        V = torch.from_numpy(V).float().to(dev)
        ones = torch.from_numpy(ones).float().to(dev)

    for i in range(0, iter_WH):
        if i % 1000 == 0: print("iter_WH: {} / {}".format(i, iter_WH))
        # update H
        if use_gpu:
            H = H * (W.transpose(0, 1).matmul(V)) / (W.transpose(0, 1).matmul(W).matmul(H) + lamda)
        else:
            H = H * (W.T.dot(V)) / (W.T.dot(W).dot(H) + lamda)  # r3n, r3r, rrn

        # update W
        Ws = W[:, 0:1]
        Wd = W[:, 1:]
        Hs = H[0].reshape(1, n)
        Hd = H[1:, :]

        Is = W[:, 0].reshape(3, 1)

        if use_gpu:
            V2 = V - Is.matmul(Hs)  # 31n
            Hd_HdT = Hd.matmul(Hd.transpose(0, 1))  # (r-1)n(r-1)
            V2_HdT = V2.matmul(Hd.transpose(0, 1))  # 3n(r-1)
            W1 = V2_HdT + Wd * ones.matmul(Wd).matmul(Hd_HdT)  # 33(r-1), 3(r-1)(r-1)
            W2 = Wd.matmul(Hd_HdT) + Wd * ones.matmul(V2_HdT)  # 3(r-1)(r-1), 33(r-1)
            Wd = Wd * (W1 / W2)
            W = torch.cat([Ws, Wd], dim=1)
            W = W / (torch.sum(W ** 2, axis=0) ** 0.5 + 1e-9)
        else:
            V2 = V - Is.dot(Hs)  # 31n
            Hd_HdT = Hd.dot(Hd.T)  # (r-1)n(r-1)
            V2_HdT = V2.dot(Hd.T)  # 3n(r-1)
            W1 = V2_HdT + Wd * ones.dot(Wd).dot(Hd_HdT)  # 33(r-1), 3(r-1)(r-1)
            W2 = Wd.dot(Hd_HdT) + Wd * ones.dot(V2_HdT)  # 3(r-1)(r-1), 33(r-1)
            Wd = Wd * (W1 / W2)
            W = np.hstack((Ws, Wd))
            W = W / (np.sum(W ** 2, axis=0) ** 0.5 + 1e-9)
            W = W.astype('float32')

    for i in range(0, iter_H):
        if i % 1000 == 0: print("iter_H: {} / {}".format(i, iter_H))
        if use_gpu:
            H = H * (W.transpose(0, 1).matmul(V)) / (W.transpose(0, 1).matmul(W).matmul(H) + lamda)
        else:
            H = H * (W.T.dot(V)) / (W.T.dot(W).dot(H) + lamda)  # r3n, r3r, rrn

    if use_gpu:
        W = W.cpu().data.numpy()
        H = H.cpu().data.numpy()

    Is = W[:, 0].reshape(3, 1)
    Hs = H[0].reshape(1, n)

    img_s = np.require(np.zeros((h, w, 3)), dtype='float32', requirements='C')
    img_s = img_s.reshape((-1, 3))
    img_s = Is.dot(Hs).T
    img_s = img_s.clip(0, 255)
    img_s = img_s.astype('uint8')
    return img_s


if __name__ == '__main__':
    save_dir = "nmf_results"
    img_path = "test_imgs/cups.png"
    img = cv2.imread(img_path).astype(np.float32)  # H, W, C
    shape = img.shape
    # img = cv2.resize(img, shape)

    illum_color = [1., 1., 1.]
    iter = 5000
    lamda = 3
    r = 7
    hlt = nmf(img, illum_color, lamda, r, iter, iter, use_gpu=torch.cuda.is_available())
    hlt = np.clip(hlt.reshape(shape), a_min=0, a_max=255).astype(np.uint8).reshape(shape)
    # hlt = (hlt - hlt.min()) / (hlt.max() - hlt.min())
    de_hlt = img - hlt
    cv2.imwrite(os.path.join(save_dir, "hlt.png"), hlt)
    cv2.imwrite(os.path.join(save_dir, "de_hlt.png"), de_hlt)
