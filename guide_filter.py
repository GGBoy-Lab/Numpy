# -*- coding: utf-8 -*-
import cv2
import numpy as np

import numpy as np


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


def _validate_inputs(I, P, radius, step):
    """
    验证输入参数的有效性

    Parameters:
    ----------
    I: 引导图像
    P: 输入图像
    radius: 滤波器半径
    step: 下采样的步长

    Raises:
    ------
    ValueError: 如果参数无效，则抛出异常
    """
    if radius <= 0 or step <= 0:
        raise ValueError("滤波器半径和下采样步长必须大于 0")
    if I.shape[:2] != P.shape[:2]:
        raise ValueError("引导图像和输入图像的尺寸必须一致")
    if I.size == 0 or P.size == 0:
        raise ValueError("输入图像不能为空")


def guide_filter_gray(I, P, radius, step, eps):
    """
    快速引导滤波 (Fast Guide Filter)，适用于灰度引导图像

    Parameters:
    ----------
    I: 灰度引导图像 (单通道)
    P: 输入图像，可以是灰度或彩色
    radius: 盒式滤波器的半径
    step: 下采样的步长
    eps: 正则化因子

    Returns:
    -------
    result: 经过引导滤波后的图像
    """
    _validate_inputs(I, P, radius, step)

    I = np.squeeze(I).astype(np.float32)
    P = np.squeeze(P).astype(np.float32)

    if I.ndim != 2:
        raise ValueError("引导图像必须是灰度图像。")

    original_data_type = P.dtype
    height, width = I.shape[:2]
    down_size = (width // step, height // step)

    # 下采样
    I_down = cv2.resize(I, dsize=down_size, interpolation=cv2.INTER_NEAREST)
    P_down = cv2.resize(P, dsize=down_size, interpolation=cv2.INTER_NEAREST)
    radius_down = max(1, radius // step)

    # 计算均值和协方差
    mean_I = boxfilter(I_down, radius_down)
    mean_P = boxfilter(P_down, radius_down)
    corr_I = boxfilter(I_down * I_down, radius_down)
    corr_IP = boxfilter(I_down * P_down, radius_down)

    var_I = corr_I - mean_I * mean_I
    cov_IP = corr_IP - mean_I * mean_P

    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I

    # 平滑系数
    mean_a = boxfilter(a, radius_down)
    mean_b = boxfilter(b, radius_down)

    # 上采样
    mean_a_up = cv2.resize(mean_a, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    mean_b_up = cv2.resize(mean_b, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    result = mean_a_up * I + mean_b_up
    #result = mean_a_up * I*1.5 + mean_b_up    #亮度增强
    #result = (I - result)*2 + result    # 细节增强

    # 后处理数据类型
    if original_data_type == np.uint8:
        result = np.clip(np.round(result), 0, 255).astype(np.uint8)
    return result


def guide_filter_color(I, P, radius, step, eps):
    """
    快速引导滤波 (Fast Guide Filter)，适用于彩色引导图像

    Parameters:
    ----------
    I: 彩色引导图像 (3 通道)
    P: 输入图像，可以是灰度或彩色
    radius: 盒式滤波器的半径
    step: 下采样的步长
    eps: 正则化因子

    Returns:
    -------
    result: 经过引导滤波后的图像
    """
    _validate_inputs(I, P, radius, step)

    I = np.squeeze(I).astype(np.float32)
    P = np.squeeze(P).astype(np.float32)

    if I.ndim < 3 or I.shape[2] != 3:
        raise ValueError("引导图像必须有 3 个通道。")

    original_data_type = P.dtype
    height, width = I.shape[:2]
    down_size = (width // step, height // step)

    # 下采样
    I_down = cv2.resize(I, dsize=down_size, interpolation=cv2.INTER_NEAREST)
    P_down = cv2.resize(P, dsize=down_size, interpolation=cv2.INTER_NEAREST)
    radius_down = max(1, radius // step)

    # 计算均值和协方差矩阵
    mean_I = boxfilter(I_down, radius_down)

    var_I_00 = boxfilter(I_down[..., 0] * I_down[..., 0], radius_down) - \
               mean_I[..., 0] * mean_I[..., 0] + eps
    var_I_11 = boxfilter(I_down[..., 1] * I_down[..., 1], radius_down) - \
               mean_I[..., 1] * mean_I[..., 1] + eps
    var_I_22 = boxfilter(I_down[..., 2] * I_down[..., 2], radius_down) - \
               mean_I[..., 2] * mean_I[..., 2] + eps
    var_I_01 = boxfilter(I_down[..., 0] * I_down[..., 1], radius_down) - \
               mean_I[..., 0] * mean_I[..., 1]
    var_I_02 = boxfilter(I_down[..., 0] * I_down[..., 2], radius_down) - \
               mean_I[..., 0] * mean_I[..., 2]
    var_I_12 = boxfilter(I_down[..., 1] * I_down[..., 2], radius_down) - \
               mean_I[..., 1] * mean_I[..., 2]

    inv_00 = var_I_11 * var_I_22 - var_I_12 * var_I_12
    inv_11 = var_I_00 * var_I_22 - var_I_02 * var_I_02
    inv_22 = var_I_00 * var_I_11 - var_I_01 * var_I_01
    inv_01 = var_I_02 * var_I_12 - var_I_01 * var_I_22
    inv_02 = var_I_01 * var_I_12 - var_I_02 * var_I_11
    inv_12 = var_I_02 * var_I_01 - var_I_00 * var_I_12

    det = var_I_00 * inv_00 + var_I_01 * inv_01 + var_I_02 * inv_02

    inv_00 /= det
    inv_11 /= det
    inv_22 /= det
    inv_01 /= det
    inv_02 /= det
    inv_12 /= det

    # 对每个通道进行滤波
    mean_P = boxfilter(P_down, radius_down)
    channels = min(3, mean_P.shape[2]) if P.ndim >= 3 else 1
    result = np.zeros_like(P, dtype=np.float32)

    for ch in range(channels):
        mean_P_channel = mean_P[..., ch]
        P_channel = P_down[..., ch] if P.ndim >= 3 else P_down
        mean_Ip = boxfilter(I_down * P_channel[..., None], radius_down)
        cov_Ip = mean_Ip - mean_I * mean_P_channel[..., None]

        a0 = inv_00 * cov_Ip[..., 0] + inv_01 * cov_Ip[..., 1] + inv_02 * cov_Ip[..., 2]
        a1 = inv_01 * cov_Ip[..., 0] + inv_11 * cov_Ip[..., 1] + inv_12 * cov_Ip[..., 2]
        a2 = inv_02 * cov_Ip[..., 0] + inv_12 * cov_Ip[..., 1] + inv_22 * cov_Ip[..., 2]
        b = mean_P_channel - a0 * mean_I[..., 0] - a1 * mean_I[..., 1] - a2 * mean_I[..., 2]

        a = np.stack([a0, a1, a2], axis=-1)
        mean_a = boxfilter(a, radius_down)
        mean_b = boxfilter(b, radius_down)

        mean_a_up = cv2.resize(mean_a, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        mean_b_up = cv2.resize(mean_b, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

        result[..., ch] = np.sum(mean_a_up * I, axis=-1) + mean_b_up

    # 后处理数据类型
    if original_data_type == np.uint8:
        result = np.clip(np.round(result), 0, 255).astype(np.uint8)
    return result
