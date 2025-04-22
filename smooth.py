import cv2
import numpy as np
from fguidefilter import guide_filter_gray


def apply_guide_filter(image, radius, step, eps_values):
    """
    批量应用引导滤波器，减少重复计算。
    :param image: 输入图像
    :param radius: 引导滤波半径
    :param step: 引导滤波步长
    :param eps_values: 不同的 eps 值列表
    :return: 包含不同 eps 值的结果列表
    """
    results = []
    for eps in eps_values:
        filtered_image = guide_filter_gray(image, image, radius, step, eps)
        results.append(filtered_image)
    return results


def smooth_skin(input_path, output_path, xmin=0, ymin=0, xmax=320, ymax=800, scale=255, step=4, radius=16):
    """
    平滑主函数，支持动态路径和参数。
    :param input_path: 输入图像路径
    :param output_path: 输出图像路径
    :param xmin: 切片起始 x 坐标
    :param ymin: 切片起始 y 坐标
    :param xmax: 切片结束 x 坐标
    :param ymax: 切片结束 y 坐标
    :param scale: 缩放因子
    :param step: 引导滤波步长
    :param radius: 引导滤波半径
    """
    try:
        # 读取图像并检查是否成功
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {input_path}")

        # 验证切片范围是否合法
        height, width = image.shape[:2]
        if not (0 <= ymin < ymax <= height and 0 <= xmin < xmax <= width):
            raise ValueError("切片范围超出图像尺寸")

        # 定义不同的 eps 值
        eps_values = [0.02, 0.04, 0.06, 0.08, 0.1]
        eps_values = [eps * eps * scale * scale for eps in eps_values]

        # 批量应用引导滤波
        filtered_images = apply_guide_filter(image, radius, step, eps_values)

        # 切片操作
        image_cropped = image[ymin:ymax, xmin:xmax]
        filtered_images_cropped = [img[ymin:ymax, xmin:xmax] for img in filtered_images]

        # 拼接显示图像
        image_up = np.concatenate([image_cropped, filtered_images_cropped[0], filtered_images_cropped[1]], axis=1)
        image_down = np.concatenate([filtered_images_cropped[2], filtered_images_cropped[3], filtered_images_cropped[4]], axis=1)
        image_all = np.concatenate([image_up, image_down], axis=0)

        # 保存结果
        cv2.imwrite(output_path, image_all)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    # 动态路径示例
    input_path = "./data/11.png"
    output_path = "smooth_box_filter.jpg"
    smooth_skin(input_path, output_path)
