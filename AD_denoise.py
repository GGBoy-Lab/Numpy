import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# 配置参数
image_file = '../images/10.png'  # 输入图像路径
iterations =5  # 迭代次数
delta = 0.14  # 更新步长
kappa = 10  # 减小kappa以增强边缘响应
alpha = 0.2  # 梯度叠加系数

# 读取并转换输入图像为浮点类型
im = plt.imread(image_file)
if im.ndim == 3:  # 如果是彩色图像，转为灰度
    im = np.mean(im, axis=2)
im = im.astype('float32') / 255.0  # 归一化到[0,1]

# 初始化条件
u = im.copy()

# 修正后的有限差分核（8个方向）
windows = [
    # 主方向
    np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float32),  # 北
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float32),  # 南
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32),  # 东
    np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float32),  # 西
    # 对角线方向
    np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float32),  # 东北
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float32),  # 西北
    np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32),  # 东南
    np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float32)  # 西南
]

# 正确的空间步长
dx = dy = 1.0
dd = np.sqrt(2)

for _ in range(iterations):
    # 计算各方向梯度
    nabla = [ndimage.convolve(u, w) for w in windows]

    # 计算扩散系数（反向响应增强边缘）
    diff = [1.0 / (1 + (np.abs(n) / kappa) ** 2) for n in nabla]

    # 组合扩散项（主方向权重1，对角线权重1/2）
    terms = [
        *[diff[i] * nabla[i] for i in range(4)],  # 主方向
        *[(1 / dd ** 2) * diff[i] * nabla[i] for i in range(4, 8)]  # 对角线
    ]

    # 更新图像
    u += delta * np.sum(terms, axis=0)

# 计算最终梯度并增强
Ix = ndimage.convolve(u, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
Iy = ndimage.convolve(u, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
G = np.sqrt(Ix ** 2 + Iy ** 2)
G_norm = G / np.max(G)  # 归一化梯度

# 梯度增强（原图与扩散后图像的加权组合）
enhanced = np.clip(u + alpha * G, 0, 1)

# 可视化
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1).imshow(im, cmap='gray'), plt.title('Original')
plt.subplot(1, 4, 2).imshow(u, cmap='gray'), plt.title('Diffused')
plt.subplot(1, 4, 3).imshow(G_norm, cmap='gray'), plt.title('Gradient')
plt.subplot(1, 4, 4).imshow(enhanced, cmap='gray'), plt.title('Enhanced')
plt.tight_layout()
plt.show()
