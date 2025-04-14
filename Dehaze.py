import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte

def validate_image(img):
    """校验输入图像是否为灰度图像"""
    if img is None or img.size == 0:
        raise ValueError("输入图像不能为空")
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("输入图像必须是灰度图像且类型为uint8")


def safe_uint8_conversion(img):
    """安全转换为uint8格式"""
    if img.dtype == np.uint8:
        return img.copy()
    # 归一化到0-255范围后转换
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    return img_as_ubyte(img_normalized)

def morphological_close(img, kernel_size=5):
    """形态学闭运算实现"""
    img_uint = safe_uint8_conversion(img)
    kernel = disk(kernel_size // 2)

    # 执行形态学操作
    dilated = rank.mean(img_uint, kernel)
    closed = rank.mean(dilated, kernel)

    # 保持浮点类型后续处理
    return closed.astype(np.float32) if img.dtype == np.float32 else closed


def quadtree_segmentation(img, min_size=16):
    """
    使用四叉树递归分割图像，直到图像的每个区域都不大于最小尺寸或方差最小。

    参数:
    img: 输入的图像，通常是一个二维数组。
    min_size: 最小分割尺寸，默认值为16，当图像的行或列小于等于这个值时，不再进行分割。

    返回:
    返回分割后的图像区域。
    """
    # 获取图像的行数和列数
    rows, cols = img.shape

    # 如果图像的行数或列数小于等于最小尺寸，则直接返回图像，不再进行分割
    if rows <= min_size or cols <= min_size:
        return img

    # 计算当前图像区域的均值和方差，用于判断是否需要进一步分割
    # mean = np.mean(img)
    # variance = np.var(img)

    # 将图像分割为四个区域：左上、右上、左下、右下
    half_row, half_col = rows // 2, cols // 2
    top_left = img[:half_row, :half_col]
    top_right = img[:half_row, half_col:]
    bottom_left = img[half_row:, :half_col]
    bottom_right = img[half_row:, half_col:]

    # 对四个区域进行递归分割
    segments = [top_left, top_right, bottom_left, bottom_right]
    variances = [np.var(segment) for segment in segments]

    # 选择方差最小的区域进行进一步分割
    min_var_idx = np.argmin(variances)
    segments[min_var_idx] = quadtree_segmentation(segments[min_var_idx], min_size)

    # 返回方差最小且进一步分割后的图像区域
    return segments[min_var_idx]


def estimate_atmospheric_light(img, candidate_region):
    """估计大气光"""
    # 在候选区域中取亮度前0.1%像素的中值
    flat_region = candidate_region.flatten()
    sorted_region = np.sort(flat_region)
    top_0_1_percent = sorted_region[int(0.999 * len(sorted_region)):]
    atmospheric_light = np.median(top_0_1_percent)
    return atmospheric_light

def boxfilter(img, radius):
    """高效盒式滤波器实现"""
    rows, cols = img.shape
    im_cum = np.cumsum(img, axis=0)

    # y方向滤波
    dst = np.zeros_like(img)
    dst[0:radius + 1, :] = im_cum[radius:2 * radius + 1, :]
    dst[radius + 1:rows - radius, :] = im_cum[2 * radius + 1:rows, :] - im_cum[0:rows - 2 * radius - 1, :]
    dst[rows - radius:rows, :] = np.tile(im_cum[rows - 1:rows, :], (radius, 1)) - im_cum[
                                                                                  rows - 2 * radius - 1:rows - radius - 1,
                                                                                  :]

    # x方向滤波
    im_cum = np.cumsum(dst, axis=1)
    dst[:, 0:radius + 1] = im_cum[:, radius:2 * radius + 1]
    dst[:, radius + 1:cols - radius] = im_cum[:, 2 * radius + 1:cols] - im_cum[:, 0:cols - 2 * radius - 1]
    dst[:, cols - radius:cols] = np.tile(im_cum[:, cols - 1:cols], (1, radius)) - im_cum[:,
                                                                                  cols - 2 * radius - 1:cols - radius - 1]

    return dst


def guided_filter(I, p, radius=1, eps=0.05):
    """
    I: 引导图像（灰度，0-255）
    p: 输入图像（需滤波图像）
    radius: 滤波半径
    eps: 正则化参数
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    # 计算局部统计量
    mean_I = boxfilter(I, radius) / (radius * 2 + 1) ** 2
    mean_p = boxfilter(p, radius) / (radius * 2 + 1) ** 2
    corr_I = boxfilter(I * I, radius) / (radius * 2 + 1) ** 2
    corr_Ip = boxfilter(I * p, radius) / (radius * 2 + 1) ** 2

    # 计算方差和协方差
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    # 计算线性系数
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 计算系数均值
    mean_a = boxfilter(a, radius) / (radius * 2 + 1) ** 2
    mean_b = boxfilter(b, radius) / (radius * 2 + 1) ** 2

    # 生成输出图像
    q = mean_a * I + mean_b
    return np.clip(q, 0, 255).astype(np.uint8)

def dehaze_ultrasound_optimized(img, omega=0.3, kernel_size=3, min_size=8, beta=0.5):
    """优化后的暗通道去雾"""
    denoised = np.clip(img, 0, 255).astype(np.uint8)

    # 优化暗通道计算
    dark_channel = morphological_close(denoised, kernel_size)

    # 大气光估计
    candidate_region = quadtree_segmentation(denoised, min_size)
    atmospheric = estimate_atmospheric_light(denoised, candidate_region)

    # 动态下限 t0 计算
    S_x = dark_channel.astype(float) / (atmospheric + 1e-6)
    t0 = 0.1 + 0.2 * (1 - S_x)

    # 透射率计算
    transmission = 1 - omega * S_x
    transmission = np.clip(transmission, t0, 1)

    # 透射率修正
    t_corrected = np.where(transmission < t0, t0 + beta * (t0 - transmission), transmission)

    # 引导滤波优化
    transmission_refined = guided_filter(
        denoised.astype(np.uint8),
        (t_corrected * 255).astype(np.uint8)
    )
    transmission_refined = transmission_refined.astype(float) / 255

    # 图像恢复
    recovered = (denoised.astype(float) - atmospheric) / (transmission_refined + 1e-6) + atmospheric
    return np.clip(recovered, 0, 255).astype(np.uint8)



# 主流程保持不变
if __name__ == "__main__":
    try:
        # 1. 读取图像
        img = plt.imread('./data/11.png')

        # 2. 转换为灰度图并归一化
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = (img * 255).astype(np.uint8)  # 确保初始为uint8

        validate_image(img)
        # 1. 暗通道去雾
        dehazed = dehaze_ultrasound_optimized(img, omega=0.17, kernel_size=9, min_size=16)

        #  结果显示
        plt.figure(figsize=(18, 12))
        plt.subplot(121), plt.imshow(img, cmap='gray', aspect='auto'), plt.title('ori_image')
        plt.subplot(122), plt.imshow(dehazed, cmap='gray', aspect='auto'), plt.title('dehaze_image')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"程序运行出错: {e}")

