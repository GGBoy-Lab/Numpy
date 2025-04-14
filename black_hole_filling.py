import numpy as np
from scipy.signal import convolve2d
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
def safe_uint8_conversion(img):
    """安全转换为uint8格式"""
    if img.dtype == np.uint8:
        return img.copy()
    # 归一化到0-255范围后转换
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    return img_as_ubyte(img_normalized)


def safe_uint8_conversion(img):
    """确保图像数据类型为uint8"""
    return np.clip(img, 0, 255).astype(np.uint8)

def black_hole_filling(img, kernel_size=5, low_th=7, high_th=15):
    """
    黑洞填充函数，用于对图像中的黑洞区域进行平滑处理。

    参数:
        img (numpy.ndarray): 输入的灰度图像，数据类型可以是任意数值类型。
        kernel_size (int): 卷积核的大小，默认为5。较大的值会导致更强的平滑效果。
        low_th (int): 低阈值，默认为7。用于定义差异值小于该阈值时的处理方式。
        high_th (int): 高阈值，默认为15。用于定义差异值大于该阈值时的处理方式。

    返回值:
        numpy.ndarray: 处理后的图像，数据类型为uint8。

    功能描述:
        该函数通过卷积操作计算图像的水平和垂直平均值，并根据像素与平均值的差异来决定如何填充像素值。
        具体步骤如下：
    """
    # 确保输入图像是uint8格式，以便后续处理
    img_uint = safe_uint8_conversion(img)

    # 创建垂直和水平的卷积核，用于计算图像的平滑值
    h = np.ones((kernel_size, 1)) / kernel_size  # 垂直卷积核
    g = np.ones((1, kernel_size)) / kernel_size  # 水平卷积核

    # 使用卷积核对图像进行平滑处理，mode='same'表示输出图像大小与输入相同
    a = convolve2d(img_uint, h, mode='same')  # 垂直方向平滑
    b = convolve2d(img_uint, g, mode='same')  # 水平方向平滑

    # 计算垂直和水平平滑结果的平均值
    avg = (a + b) / 2

    # 计算原始图像与平均值的差异
    diff = img_uint - avg

    # 根据差异值与阈值的关系，定义不同的掩码
    mask_low = diff < -low_th  # 差异小于负低阈值的区域
    mask_high = diff > high_th  # 差异大于高阈值的区域
    mask_mid = np.logical_and(diff >= -low_th, diff <= high_th)  # 差异在低阈值和高阈值之间的区域

    # 根据掩码计算每个像素的填充值
    replacement = np.where(mask_low, avg, img_uint)  # 差异过小的区域用平均值替换
    replacement = np.where(mask_high, img_uint, replacement)  # 差异过大的区域保持原值
    replacement = np.where(mask_mid,
                           avg + (img_uint - avg) * (diff + low_th) / (high_th - low_th),
                           replacement)  # 差异适中的区域按比例调整

    # 返回处理后的图像，确保数据类型为uint8
    return replacement.astype(np.uint8)



if __name__ == "__main__":
    try:
        # 1. 读取图像
        img = plt.imread('./data/10.png')

        # 2. 转换为灰度图并归一化
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = (img * 255).astype(np.uint8)  # 确保初始为uint8

        # 3. 应用黑洞填充
        img_filled = black_hole_filling(img)
        plt.figure(figsize=(18, 12))
        plt.subplot(121), plt.imshow(img, cmap='gray', aspect='auto'), plt.title('ori_image')
        plt.subplot(122), plt.imshow(img_filled, cmap='gray', aspect='auto'), plt.title('filled_image')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"程序运行出错: {e}")

