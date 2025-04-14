import numpy as np
import scipy as sp
import scipy.ndimage


def box(img, r):
    """ O(1) 箱式滤波器 (Box Filter)
        img - 输入图像，至少为二维
        r   - 箱式滤波器的半径
    """
    (rows, cols) = img.shape[:2]  # 获取图像的高度和宽度
    imDst = np.zeros_like(img)  # 创建一个与输入图像相同大小的零矩阵

    # 沿着行方向进行累积求和
    tile = [1] * img.ndim  # 创建一个与图像维度匹配的平铺参数
    tile[0] = r  # 设置行方向的平铺大小
    imCum = np.cumsum(img, 0)  # 按行累积求和
    imDst[0:r + 1, :, ...] = imCum[r:2 * r + 1, :, ...]  # 处理顶部区域
    imDst[r + 1:rows - r, :, ...] = imCum[2 * r + 1:rows, :, ...] - imCum[0:rows - 2 * r - 1, :, ...]  # 中间区域
    imDst[rows - r:rows, :, ...] = np.tile(imCum[rows - 1:rows, :, ...], tile) - imCum[rows - 2 * r - 1:rows - r - 1, :,
                                                                                 ...]  # 底部区域

    # 沿着列方向进行累积求和
    tile = [1] * img.ndim  # 重新设置平铺参数
    tile[1] = r  # 设置列方向的平铺大小
    imCum = np.cumsum(imDst, 1)  # 按列累积求和
    imDst[:, 0:r + 1, ...] = imCum[:, r:2 * r + 1, ...]  # 处理左侧区域
    imDst[:, r + 1:cols - r, ...] = imCum[:, 2 * r + 1: cols, ...] - imCum[:, 0: cols - 2 * r - 1, ...]  # 中间区域
    imDst[:, cols - r: cols, ...] = np.tile(imCum[:, cols - 1:cols, ...], tile) - imCum[:,
                                                                                  cols - 2 * r - 1: cols - r - 1, ...]  # 右侧区域

    return imDst


def _gf_gray(I, p, r, eps, s=None):
    """ 灰度（快速）引导滤波器 (Gray Guided Filter)
        I - 引导图像（单通道）
        p - 过滤输入（单通道）
        r - 窗口半径
        eps - 正则化参数（大致为非边缘噪声的方差）
        s - 快速引导滤波的子采样因子
    """
    if s is not None:  # 如果设置了子采样因子
        Isub = sp.ndimage.zoom(I, 1 / s, order=1)  # 对引导图像进行缩放
        Psub = sp.ndimage.zoom(p, 1 / s, order=1)  # 对过滤输入进行缩放
        r = round(r / s)  # 调整窗口半径
    else:
        Isub = I  # 使用原始引导图像
        Psub = p  # 使用原始过滤输入

    (rows, cols) = Isub.shape  # 获取引导图像的高度和宽度

    N = box(np.ones([rows, cols]), r)  # 计算窗口内的像素数量

    # 计算引导图像和过滤输入的均值
    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N

    # 计算引导图像的平方均值和引导图像与过滤输入的乘积均值
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N

    # 计算引导图像的方差和协方差
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)  # 计算系数
    b = meanP - a * meanI  # 计算偏置项

    meanA = box(a, r) / N  # 平滑系数
    meanB = box(b, r) / N  # 平滑偏置项

    if s is not None:  # 如果设置了子采样因子
        meanA = sp.ndimage.zoom(meanA, s, order=1)  # 对系数进行缩放
        meanB = sp.ndimage.zoom(meanB, s, order=1)  # 对偏置项进行缩放

    q = meanA * I + meanB  # 计算最终输出
    return q


def guided_filter(I, p, r, eps, s=None):
    """ 在过滤输入的每个通道上运行引导滤波器
        I - 引导图像（1 或 3 通道）
        p - 过滤输入（n 通道）
        r - 窗口半径
        eps - 正则化参数（大致为非边缘噪声的方差）
        s - 快速引导滤波的子采样因子
    """
    if p.ndim == 2:  # 如果过滤输入是二维图像
        p3 = p[:, :, np.newaxis]  # 添加一个通道维度
    else:
        p3 = p

    # 将输入图像转换为 float32 类型
    I = I.astype(np.float32) / 255
    p3 = p3.astype(np.float32) / 255

    out = np.zeros_like(p3)  # 初始化输出图像
    for ch in range(p3.shape[2]):  # 遍历每个通道
        out[:, :, ch] = _gf_gray(I, p3[:, :, ch], r, eps, s)  # 对每个通道应用引导滤波器

    # 将输出图像转换回 int8 类型
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return np.squeeze(out) if p.ndim == 2 else out  # 返回处理后的图像


def test_gf():
    """ 测试引导滤波器 """
    import imageio
    cat = imageio.imread('10.png')  #
    tulips = imageio.imread('10.png')  #

    r = 8  # 设置窗口半径
    eps = 0.05  # 设置正则化参数

    cat_smoothed = guided_filter(cat, cat, r, eps)  #
    cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)  #

    imageio.imwrite('cat_smoothed.png', cat_smoothed)  #
    imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)  #

    tulips_smoothed4s = np.zeros_like(tulips)  #
    for i in range(3):  #
        tulips_smoothed4s[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps, s=4)
    imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)  #

    tulips_smoothed = np.zeros_like(tulips)  #
    for i in range(3):  #
        tulips_smoothed[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps)
    imageio.imwrite('tulips_smoothed.png', tulips_smoothed)  #
