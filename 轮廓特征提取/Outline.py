import os
import numpy as np
import cv2

#轮廓是图像中目标的边界信息，常用于形状分析和物体识别。OpenCV提供了findContours函数实现轮廓提取。



#region
#读取灰度图像：cv2.imread：添加了中文路径支持（使用 cv2.imdecode + np.fromfile 替代 cv2.imread ）
path = r"d:\2024年上半年作业\机器视觉学习\区域特征提取技术\IMG_8073.JPG"
if not os.path.exists(path):
    raise FileNotFoundError(f"图像文件不存在: {path}")
gray_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    raise FileNotFoundError("无法读取图像文件，请检查路径是否正确")
if len(gray_image.shape) != 2:
    raise ValueError("输入图像必须是二维灰度图像")
#endregion

#二值化图像
# 获取阈值处理后的二值图像（取返回元组的第二个元素）
_, Binarization = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

#检测轮廓
contours,hierarchy=cv2.findContours(Binarization,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#这里可以同时赋予俩个变量

#ctrl+/:多行注释
# 同时赋予两个变量是因为 cv2.findContours() 这个函数返回了 两个有用的数据，分别是：
# 轮廓数据（contours）：它是一个包含所有检测到的轮廓的列表，每个轮廓由若干个点的坐标组成。
# 层级数据（hierarchy）：它是一个数组，表示轮廓之间的层级关系。即哪些轮廓是嵌套的，哪些是外部轮廓等。
#可视化轮廓
# 转换为彩色图像以显示彩色轮廓
color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(color_image, contours, -1, (0,255,0), 2)
cv2.imshow('Contours', color_image)
cv2.waitKey(0)  # 等待用户按键（0表示无限等待）
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


