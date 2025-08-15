
import os
import numpy as np
import cv2
from skimage.feature.texture import graycomatrix, graycoprops

#灰度值特征通过分析图像的明暗分布，捕捉目标的形状和纹理信息。灰度共生矩阵（GLCM）和局部二值模式（LBP）是常用方法。
#例子：GLCM纹理特征提取
#技术要点：GLCM：通过像素对的空间关系计算纹理特征，适用于木材、岩石等材质分类。
#LBP：捕捉局部纹理模式，对光照变化具有鲁棒性，常用于人脸识别和工业缺陷检测。


#读取灰度图像：cv2.imread：添加了中文路径支持（使用 cv2.imdecode + np.fromfile 替代 cv2.imread ）
path = r"d:\2024年上半年作业\机器视觉学习\区域特征提取技术\IMG_8073.JPG"
if not os.path.exists(path):
    raise FileNotFoundError(f"图像文件不存在: {path}")
gray_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    raise FileNotFoundError("无法读取图像文件，请检查路径是否正确")
if len(gray_image.shape) != 2:
    raise ValueError("输入图像必须是二维灰度图像")

#计算GLCM：greycomatrix

a=graycomatrix(gray_image,distances=[1],angles=[0],levels=256,symmetric=True,normed=True)#计算灰度共生矩阵

#提取对比度，相关性等统计量 greycoprops
contrast=graycoprops(a,'contrast')#提取对比度
correlation=graycoprops(a,'correlation')#提取相关性

#最后打印出来
print(f"对比度为：{contrast}  相关性为：{correlation}")

