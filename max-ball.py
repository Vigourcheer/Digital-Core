import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import medial_axis, skeletonize
from skimage import measure
from scipy import ndimage

def advanced_binarization(image_path):
    """
    高级二值化方法组合
    :param image_path: 输入图像路径
    :return: 优化后的二值图像
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像，请检查路径是否正确")
    
    # 1. 自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Otsu阈值
    _, otsu_thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. 结合两种方法
    combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
    
    # 4. 形态学处理
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    return processed

def maximal_ball_transform(binary_img):
    """
    最大球变换(骨架化+距离变换)
    :param binary_img: 二值图像
    :return: 距离变换结果和骨架
    """
    # 计算距离变换
    distance_map = ndimage.distance_transform_edt(binary_img)
    
    # 计算骨架
    skeleton = skeletonize(binary_img // 255)
    
    return distance_map, skeleton

def reconstruct_3d_structure(distance_map, skeleton):
    """
    基于最大球法的3D重构
    :param distance_map: 距离变换图
    :param skeleton: 骨架图
    :return: 3D点云数据
    """
    # 获取骨架点的坐标
    skeleton_points = np.argwhere(skeleton)
    
    # 获取这些点的半径(距离变换值)
    radii = distance_map[skeleton[:,0], skeleton[:,1]]
    
    # 创建3D点云
    points = []
    for (y, x), r in zip(skeleton_points, radii):
        # 为每个骨架点创建球体点
        if r > 0:  # 只处理有半径的点
            # 简化为在骨架点位置使用半径作为z值
            points.append([x, y, r])
    
    return np.array(points)

def visualize_3d(points):
    """
    可视化3D点云
    :param points: 3D点云数据(Nx3)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c=points[:,2], cmap='viridis', s=5)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 设置标签
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Structure Radius')
    ax.set_title('3D Reconstruction using Maximal Ball Method')
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 图像路径 - 替换为您自己的图像路径
    image_path = "example.jpg"  # 请确保图像存在
    
    try:
        # 1. 高级二值化
        binary_img = advanced_binarization(image_path)
        
        # 显示二值化图像
        cv2.imshow("Advanced Binary Image", binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 2. 最大球变换
        distance_map, skeleton = maximal_ball_transform(binary_img)
        
        # 显示距离变换和骨架
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(distance_map, cmap='jet')
        plt.title("Distance Transform")
        plt.colorbar()
        
        plt.subplot(122)
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeleton")
        plt.show()
        
        # 3. 3D重构
        points_3d = reconstruct_3d_structure(distance_map, skeleton)
        
        # 4. 可视化
        visualize_3d(points_3d)
        
    except Exception as e:
        print(f"发生错误: {e}")