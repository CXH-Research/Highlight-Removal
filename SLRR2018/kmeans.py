# code=utf-8
"""
For small data.
The speed is :
numpy > torch_cpu > torch_gpu

"""


def kmeans_numpy(ds, k, iter=500):
    # (N, 2)
    import numpy as np

    N, M = ds.shape
    result = np.empty(N)
    cores = ds[np.random.choice(np.arange(N), k, replace=False)]
    for i in range(iter):
        # (N,1,M) - (k, M)
        distance = np.linalg.norm(ds[:, np.newaxis, :] - cores, axis=2)  # N, k
        min_index = np.argmin(distance, axis=1)
        if (min_index == result).all():
            return result, cores
        result[:] = min_index
        for i in range(k):
            k_points = ds[min_index == i]  # (N, M)
            cores[i] = np.mean(k_points, axis=0)
    return result, cores

def kmeans_torch(ds, k, iter=500, use_cuda=False):
    import numpy as np
    import torch
    N, M = ds.shape
    ds = torch.from_numpy(ds)
    result = torch.empty(N)
    if use_cuda and torch.cuda.is_available():
        ds = ds.cuda()
        result = result.cuda()
        print("Use GPU")
    else:
        print("Use CPU")

    cores = ds[np.random.choice(np.arange(N), k, replace=False)]
    for i in range(iter):
        # (N,1,M) - (k, M)
        distance = torch.norm(ds[:, np.newaxis, :] - cores, dim=2)  # N, k
        min_index = torch.argmin(distance, dim=1)
        if (min_index == result).all():
            result = result.cpu().numpy()
            cores = cores.cpu().numpy()
            return result, cores
        result[:] = min_index
        for i in range(k):
            k_points = ds[min_index == i]  # (N, M)
            cores[i] = torch.mean(k_points, dim=0)
    result = result.cpu().numpy()
    cores = cores.cpu().numpy()

def create_data_set(*cores):
    """生成k-means聚类测试用数据集"""
    import numpy as np
    ds = list()
    for x0, y0, z0 in cores:
        x = np.random.normal(x0, 0.1 + np.random.random() / 3, z0)
        y = np.random.normal(y0, 0.1 + np.random.random() / 3, z0)
        ds.append(np.stack((x, y), axis=1))
    return np.vstack(ds)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    k = 50
    nums = 25000
    ds = create_data_set((0, 0, nums), (0, 2, nums), (2, 0, nums), (2, 2, nums))

    t0 = time.time()
    result, cores = kmeans_numpy(ds, k)
    t = time.time() - t0

    plt.scatter(ds[:, 0], ds[:, 1], s=1, c=result.astype(np.int))
    plt.scatter(cores[:, 0], cores[:, 1], marker='x', c=np.arange(k))
    plt.show()
    print(u'使用kmeans算法，1万个样本点，耗时%f0.3秒' % t)
