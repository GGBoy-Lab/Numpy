import numpy as np
import rasterio
from rasterio.windows import Window



# 获取有效数据掩膜
def get_valid_data_mask(band_data, nodata_value=0):
    """获取有效数据掩膜，排除黑边和无效值"""
    if nodata_value is not None:
        mask = band_data != nodata_value
    else:
        # 自动检测黑边
        nonzero_mask = band_data > 0
        if np.any(nonzero_mask):
            valid_data = band_data[nonzero_mask]
            threshold = np.percentile(valid_data, 1)  # 使用1%分位数过滤
            mask = band_data > threshold
        else:
            mask = np.ones_like(band_data, dtype=bool)
    return mask


# 百分比拉伸函数
def get_percent_stretch_tile(img_tile, lower_percent=2, upper_percent=98, nodata_value=0):
    """对切片进行百分比拉伸，智能处理黑边区域"""
    res_tile = img_tile.astype(np.float32)

    for b in range(res_tile.shape[0]):
        band_data = res_tile[b]

        # 获取有效数据掩膜
        valid_mask = get_valid_data_mask(band_data, nodata_value)

        # 只对有效数据计算百分位数
        if np.any(valid_mask):
            valid_data = band_data[valid_mask]
            low = np.percentile(valid_data, lower_percent)
            high = np.percentile(valid_data, upper_percent)

            if high > low:
                # 对整个波段执行拉伸
                stretched_data = (band_data - low) * (255.0 / (high - low))
                res_tile[b] = np.clip(stretched_data, 0, 255)

    return res_tile


# 新增边界检测函数
def detect_image_bounds(src, sample_bands=1):
    """检测图像实际边界，排除黑边区域"""
    h, w = src.height, src.width
    bounds = {'top': 0, 'bottom': h, 'left': 0, 'right': w}

    # 采样检测边界
    for i in range(min(sample_bands, src.count)):
        band_data = src.read(i + 1)
        nonzero_rows = np.any(band_data > 0, axis=1)
        nonzero_cols = np.any(band_data > 0, axis=0)

        row_indices = np.where(nonzero_rows)[0]
        col_indices = np.where(nonzero_cols)[0]

        if len(row_indices) > 0:
            bounds['top'] = max(bounds['top'], row_indices[0])
            bounds['bottom'] = min(bounds['bottom'], row_indices[-1] + 1)
        if len(col_indices) > 0:
            bounds['left'] = max(bounds['left'], col_indices[0])
            bounds['right'] = min(bounds['right'], col_indices[-1] + 1)

    return bounds


def create_weight_mask(window_size):
    """创建一个中心权重高、边缘权重低的遮罩，用于无缝融合"""
    # 创建线性渐变：从边缘的 0 到中心的 1
    mask_1d = np.linspace(0, 1, window_size // 2)
    mask_1d = np.concatenate([mask_1d, mask_1d[::-1]])

    # 如果 window_size 是奇数，补齐一位
    if len(mask_1d) < window_size:
        mask_1d = np.insert(mask_1d, window_size // 2, 1.0)

    mask_2d = np.outer(mask_1d, mask_1d)
    return mask_2d.astype(np.float32)


def seamless_enhance_tif(input_path, output_path, window_size=1024, overlap=256):
    """
    全覆盖重叠滑窗增强，保留空间坐标
    :param overlap: 重叠像素宽度
    """
    stride = window_size - overlap

    with rasterio.open(input_path) as src:
        # 准备输出配置，保留 CRS 和 Transform
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=src.count, nodata=None)

        h, w = src.height, src.width
        bands = src.count

        # 初始化全局累加器和权重图 (float32 保证计算精度)
        accumulator = np.zeros((bands, h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        # 获取基础权重遮罩
        base_mask = create_weight_mask(window_size)

        # 生成滑窗坐标，确保覆盖到最后一个像素
        def get_coords(total_size, win_size, step):
            coords = list(range(0, total_size - win_size, step))
            coords.append(total_size - win_size)  # 强制加入边缘起点
            return sorted(list(set(coords)))

        y_coords = get_coords(h, window_size, stride)
        x_coords = get_coords(w, window_size, stride)

        print(f"开始处理: {w}x{h}, 窗口总数: {len(y_coords) * len(x_coords)}")

        for y in y_coords:
            for x in x_coords:
                # 1. 定义读取窗口
                rw = Window(x, y, window_size, window_size)
                tile = src.read(window=rw).astype(np.float32)

                # 2. 局部增强
                enhanced_tile = get_percent_stretch_tile(tile)

                # 3. 累加到全局
                # 注意：如果图像边缘不足一个 window_size（通常不会，因为我们往回跳了），切片大小是一致的
                accumulator[:, y:y + window_size, x:x + window_size] += enhanced_tile * base_mask
                weight_sum[y:y + window_size, x:x + window_size] += base_mask

                print(f"进度: 正在处理块 ({x}, {y})", end='\r')

        # 4. 归一化融合结果
        print("\n正在生成最终图像并注入坐标信息...")
        # 避免除以0，对于未覆盖区域（理论上没有）设为1
        weight_sum[weight_sum == 0] = 1.0

        for b in range(bands):
            accumulator[b] /= weight_sum

        final_img = np.clip(accumulator, 0, 255).astype(np.uint8)

        # 5. 写入带地理信息的 TIF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(final_img)

    print(f"成功！结果保存在: {output_path}")


if __name__ == "__main__":
    # 参数建议：window_size=1024, overlap=256 (即 1/4 重叠)
    # 每个像素至少被处理 1-4 次，边缘非常丝滑
    seamless_enhance_tif(
        r"E:\Desktop\Dehae_DataSet\WL_0820A1.tif",
        r"E:\Desktop\Dehae_DataSet\result\seamless_output4.tif",
        window_size=1024,
        overlap=512
    )