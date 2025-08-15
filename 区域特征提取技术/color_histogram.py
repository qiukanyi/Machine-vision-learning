import cv2
import numpy as np
from matplotlib import pyplot as plt
import os# 检查当前工作目录


#总是报文件错误
# 构建图像路径
image_path = os.path.join('d:\\', '2024年上半年作业', '机器视觉学习', '区域特征提取技术', 'IMG_8073.JPG')
print(f"构建的图像路径: {image_path}")

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"文件不存在: {image_path}")
    exit(1)

# 读取图像并转化为HSV颜色空间
# 使用文件流读取图像以支持中文路径
with open(image_path, 'rb') as f:#以二进制读取图像
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if image is None:
    print("无法读取图像文件，请检查文件完整性")
    exit(1)
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#region
#  核心方法1：作用就是将传入的图像从BGR颜色空间转化为HSV颜色空间
# cv2.COLOR_BGR2GRAY：将 BGR 转换为灰度图。
# cv2.COLOR_BGR2RGB：将 BGR 转换为 RGB。
# cv2.COLOR_HSV2BGR：将 HSV 转换为 BGR。
#endregion

#region
#cv2.imread() 函数的参数说明如下：

# ### 1. 必选参数：文件路径
# 第一个参数是 图像文件的路径 （字符串类型），可以是：

# - 相对路径 ：如 "input.jpg" （表示当前工作目录下的文件）
# - 绝对路径 ：如 "d:/2024年上半年作业/机器视觉学习/区域特征提取技术/IMG_8073.JPG" （完整路径，Windows系统使用反斜杠 \ 或正斜杠 / ）
# ### 2. 可选参数：读取模式（flags）
# 第二个参数是可选的读取模式，常用取值：

# - cv2.IMREAD_COLOR （默认值，可省略）：读取彩色图像，忽略透明度通道，返回 shape 为 (高度, 宽度, 3) 的 BGR 格式数组 。
# - cv2.IMREAD_GRAYSCALE ：读取为灰度图像，返回 shape 为 (高度, 宽度) 的单通道数组。
# - cv2.IMREAD_UNCHANGED ：读取完整图像（包括透明度通道），返回 shape 为 (高度, 宽度, 4) 的数组（若图像有 alpha 通道）。
#endregion

#计算颜色直方图
hist =cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])

#可视化直方图
plt.imshow(hist,interpolation='nearest')
plt.title('2D Histogram of H and S')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.colorbar()
plt.show()