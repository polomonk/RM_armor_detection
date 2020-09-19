import cv2
import numpy as np
from math import atan2, sqrt

#世界坐标系位置
object_3d_points = np.array(([-67.5, 62.5, 0],
                             [67.5, 62.5, 0],
                             [67.5, -62.5, 0],
                             [-67.5, -62.5, 0]), dtype=np.float)
#内参矩阵
camera_matrix = np.array(([383.24421089, 0, 339.51310798],
                          [0, 383.12321954, 236.42845787],
                          [0, 0, 1.0]), dtype=np.float)
#畸变矩阵
dist_coefs = np.array([0.06804571, -0.21537906, -0.00031575, 0.00251214, 0.2158873], dtype=np.float)

def pnp_pose(left_up_point,right_up_point,right_down_point,left_down_point):
    #相机坐标系位置
    object_2d_point = np.array([left_up_point,right_up_point,right_down_point,left_down_point], dtype=np.float)

    found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)

    # 输出二维码位置信息
    rotM = cv2.Rodrigues(rvec)[0]  # solvePnP返回的raux是旋转向量，可通过罗德里格斯变换成旋转矩阵R。
    camera_postion = -np.mat(rotM).T * np.mat(tvec)  # 得出相机相对现实坐标系的坐标
    # print("x,y,z轴位置：")
    # print(camera_postion.T)  # 输出为 x,y,z轴信息
    pose = camera_postion.T

    # # 反解出相机在现实坐标系在绕各个轴的旋转角度，正数表示顺时针旋转
    # theta_z = atan2(rotM[1][0], rotM[0][0]) * 57.2958;
    # theta_y = atan2(-rotM[2][0], sqrt(rotM[2][0] * rotM[2][0] + rotM[2][2] * rotM[2][2])) * 57.2958;
    # theta_x = atan2(rotM[2][1], rotM[2][2]) * 57.2958;
    # print("x,y,z轴的旋转角度:")
    # print(theta_x, theta_y, theta_z)

    return pose