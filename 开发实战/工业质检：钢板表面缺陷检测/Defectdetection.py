# 工业质检：钢板表面缺陷检测
# 结合HOG和CNN特征，实现毫米级缺陷识别。

# 效果：缺陷检测准确率98%，质量成本降低40%
# 示例：HOG特征提取


# 细胞单元与块归一化：通过分块归一化减少光照变化的影响，增强特征鲁棒性。
# 应用场景：HOG广泛应用于行人检测、车牌识别等工业场景。

from skimage.feature import hog
from skimage import exposure
import cv2
import numpy as np
import os
import sys


#先获取图像
#region
#读取图像：cv2.imread：添加了中文路径支持（使用 cv2.imdecode + np.fromfile 替代 cv2.imread ）
path = r"d:\2024年上半年作业\机器视觉学习\区域特征提取技术\IMG_8073.JPG"
if not os.path.exists(path):
    raise FileNotFoundError(f"图像文件不存在: {path}")
gray_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    raise FileNotFoundError("无法读取图像文件，请检查路径是否正确")
if len(gray_image.shape) != 2:
    raise ValueError("输入图像必须是二维灰度图像")
print(f"图像加载成功，尺寸: {gray_image.shape}", file=sys.stderr)
#endregion


#提取HOG特征
fd,gray_image=hog(gray_image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,channel_axis=None)
print(f"HOG特征提取完成，特征维度: {fd.shape}", file=sys.stderr)

#可视化HOG图像
gray_image__rescaled=exposure.rescale_intensity(gray_image,in_range=(0,10))
print('开始HOG可视化...', file=sys.stderr)
# 调整可视化图像大小以适应屏幕
hog_image_resized = cv2.resize(gray_image__rescaled, (800, 600))
cv2.imshow('HOG Features', hog_image_resized)
print('图像已显示，请按任意键关闭窗口...', file=sys.stderr)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('窗口已关闭', file=sys.stderr)
