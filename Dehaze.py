import cv2
import math
import numpy as np
import sys

# 提取全局常量
DARK_CHANNEL_SIZE = 15
OMEGA = 0.95
GUIDED_FILTER_RADIUS = 60
GUIDED_FILTER_EPSILON = 0.0001
RECOVER_MIN_TRANSMISSION = 0.2

def boxfilter(image, radius):
    """
    手动实现盒式滤波器 (Box Filter)

    Parameters:
    ----------
    image: 输入图像 (numpy数组)
    radius: 滤波器半径

    Returns:
    -------
    filtered_image: 经过盒式滤波后的图像
    """
    if radius <= 0:
        raise ValueError("滤波器半径必须大于 0")

    # 获取输入图像的尺寸
    height, width = image.shape[:2]
    ksize = 2 * radius + 1  # 计算滤波器窗口大小

    # 初始化输出图像
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # 遍历图像中的每个像素
    for y in range(height):
        for x in range(width):
            # 计算滤波器窗口的边界
            y_min = max(0, y - radius)
            y_max = min(height, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(width, x + radius + 1)

            # 提取窗口区域
            window = image[y_min:y_max, x_min:x_max]

            # 计算窗口内像素的平均值
            filtered_image[y, x] = np.mean(window)

    return filtered_image.astype(image.dtype)
def DarkChannel(im, sz=DARK_CHANNEL_SIZE):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = max(math.floor(imsz / 1000), 1)

    # 确保 darkvec 和 imvec 的 reshape 操作不会出错
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[-numpx:]  # 避免索引越界

    atmsum = np.zeros([1, 3])
    for ind in indices:
        atmsum += imvec[ind]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz=DARK_CHANNEL_SIZE):
    omega = OMEGA
    im3 = np.empty_like(im)

    # 向量化操作替代显式循环
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r=GUIDED_FILTER_RADIUS, eps=GUIDED_FILTER_EPSILON):
    mean_I = boxfilter(im, r)
    mean_p = boxfilter(p, r)
    mean_Ip = boxfilter(im * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter(im * im, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter(a, r)
    mean_b = boxfilter(b, r)

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    t = Guidedfilter(gray, et, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)
    return t


def Recover(im, t, A, tx=RECOVER_MIN_TRANSMISSION):
    res = np.empty_like(im)
    t = np.maximum(t, tx)  # 确保 t 不小于 tx

    # 向量化操作替代显式循环
    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except IndexError:
        print("使用默认图像路径")
        fn = './data/11.png'

    # 检查图像是否成功加载
    src = cv2.imread(fn)
    if src is None:
        print(f"无法读取图像: {fn}")
        sys.exit(1)

    I = src.astype('float64') / 255

    dark = DarkChannel(I)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A)

    cv2.imshow("dark", dark)
    cv2.imshow("t", t)
    cv2.imshow('I', src)
    cv2.imshow('J', J)
    cv2.imwrite("./image/J.png", J * 255)
    cv2.waitKey()
