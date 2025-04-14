import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel_1d(sigma, order=0):
    """
    生成一维高斯核或其导数

    参数:
    sigma -- 高斯函数的标准差
    order -- 导数阶数（0为原始高斯核，1为一阶导数，2为二阶导数）

    返回:
    g -- 一维高斯核或导数核
    """
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    if order == 1:  # 一阶导数
        g = -x / sigma ** 2 * g
        g /= np.abs(g).sum()  # 归一化
    elif order == 2:  # 二阶导数
        g = (x ** 2 - sigma ** 2) / sigma ** 4 * g
        g /= np.abs(g).sum()
    elif order != 0:
        raise ValueError("Order must be 0, 1, or 2")

    return g


def compute_hessian(image, sigma):
    """
    计算Hessian矩阵的分量Ixx, Ixy, Iyy

    参数:
    image -- 输入图像
    sigma -- 高斯函数的标准差

    返回:
    Ixx, Ixy, Iyy -- Hessian矩阵的分量
    """
    # 生成分离核
    kx2 = gaussian_kernel_1d(sigma, 2)
    ky0 = gaussian_kernel_1d(sigma, 0)
    kx1 = gaussian_kernel_1d(sigma, 1)
    ky1 = gaussian_kernel_1d(sigma, 1)
    ky2 = gaussian_kernel_1d(sigma, 2)
    kx0 = ky0  # 高斯核对称

    # 计算Ixx (先x方向二阶导，再y方向平滑)
    Ixx = np.apply_along_axis(lambda r: np.convolve(r, kx2, mode='same'), 1, image)
    Ixx = np.apply_along_axis(lambda c: np.convolve(c, ky0, mode='same'), 0, Ixx) * sigma ** 2

    # 计算Iyy (先y方向二阶导，再x方向平滑)
    Iyy = np.apply_along_axis(lambda c: np.convolve(c, ky2, mode='same'), 0, image)
    Iyy = np.apply_along_axis(lambda r: np.convolve(r, kx0, mode='same'), 1, Iyy) * sigma ** 2

    # 计算Ixy (x和y方向一阶导)
    Ixy = np.apply_along_axis(lambda r: np.convolve(r, kx1, mode='same'), 1, image)
    Ixy = np.apply_along_axis(lambda c: np.convolve(c, ky1, mode='same'), 0, Ixy) * sigma ** 2

    return Ixx, Ixy, Iyy


def compute_eigenvalues(Ixx, Ixy, Iyy):
    """
    计算Hessian矩阵的特征值

    参数:
    Ixx, Ixy, Iyy -- Hessian矩阵的分量

    返回:
    lambda1, lambda2 -- Hessian矩阵的特征值
    """
    # 计算迹和行列式
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2

    # 计算特征值
    sqrt_term = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
    lambda1 = (trace + sqrt_term) / 2
    lambda2 = (trace - sqrt_term) / 2

    # 按绝对值排序
    mask = np.abs(lambda1) > np.abs(lambda2)
    lambda1[mask], lambda2[mask] = lambda2[mask], lambda1[mask]

    return lambda1, lambda2


def frangi_response(lambda1, lambda2, beta=0.5, c=0.5):
    """
    计算Frangi滤波响应

    参数:
    lambda1, lambda2 -- Hessian矩阵的特征值
    beta -- 各向异性权重参数
    c -- 结构显著性权重参数

    返回:
    response -- Frangi滤波响应
    """
    # 初始化响应矩阵
    response = np.zeros_like(lambda1)

    # 仅处理lambda2 < 0的区域（假设暗背景中的）
    mask = (lambda2 < 0) & (np.abs(lambda2) > 1e-6)

    Rb = np.abs(lambda1[mask]) / np.abs(lambda2[mask])  # 各向异性比
    S = np.sqrt(lambda1[mask] ** 2 + lambda2[mask] ** 2)  # 结构显著性

    Ra = np.exp(-(Rb ** 2) / (2 * beta ** 2))  # 各向异性权重
    Ss = 1 - np.exp(-(S ** 2) / (2 * c ** 2))  # 结构显著性权重

    response[mask] = Ra * Ss
    return response


def frangi_filter(image, sigmas=[1], beta=0.5, c=0.5):
    """
    增强图像中的线状结构

    参数:
    image -- 输入图像
    sigmas -- 高斯函数的标准差列表，用于多尺度分析
    beta, c -- Frangi滤波的参数

    返回:
    response -- 增强后的图像
    """
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)  # 归一化

    response = np.zeros_like(image)
    for sigma in sigmas:
        Ixx, Ixy, Iyy = compute_hessian(image, sigma)
        lambda1, lambda2 = compute_eigenvalues(Ixx, Ixy, Iyy)
        current = frangi_response(lambda1, lambda2, beta, c)
        response = np.maximum(response, current)

    return response / response.max()  # 归一化输出


# --------------------- 主流程 ---------------------
if __name__ == "__main__":
    try:
        # 1. 读取图像
        img = plt.imread('./data/11.png')

        # 2. 转换为灰度图并归一化
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = (img * 255).astype(np.uint8)  # 确保初始为uint8

        # 1. 暗通道去雾
        enhanced = frangi_filter(img,
                                 sigmas=[1, 2, 3],  # 检测不同尺度
                                 beta=5,  # 各向异性参数
                                 c=1)

        img_normalized = img.astype(np.float32) / 255.0

        result = img_normalized * (1.5 + 0.8 * enhanced)  # 增强高响应区域
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)

        #  结果显示
        plt.figure(figsize=(18, 12))
        plt.subplot(221), plt.imshow(img, cmap='gray', aspect='auto'), plt.title('ori_image')
        plt.subplot(222), plt.imshow(enhanced, cmap='gray', aspect='auto'), plt.title('hessian_image')
        plt.subplot(223), plt.imshow(result, cmap='gray', aspect='auto'), plt.title('result_image')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"程序运行出错: {e}")
