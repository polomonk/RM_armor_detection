import cv2
import numpy as np
import math
# 现实中假定的坐标点
object_3d_points = np.array(([0, 0, 0],
                            [0, 200, 0],
                            [150, 0, 0],
                            [150, 200, 0]), dtype=np.double)
# 图像上对应的像素坐标
object_2d_point = np.array(([2985, 1688],
                            [5081, 1690],
                            [2997, 2797],
                            [5544, 2757]), dtype=np.double)
def get_objct_2d_point(img):
    return 0


camera_matrix = np.array(([6800.7, 0, 3065.8],
                         [0, 6798.1, 1667.6],
                         [0, 0, 1.0]), dtype=np.double)
dist_coefs = np.array([-0.189314, 0.444657, -0.00116176, 0.00164877, -2.57547], dtype=np.double)


from math import atan2, sqrt
def get_postion_and_angle(rotMat):

        # 求解相机位姿
    found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)
    if found:
        rotM = cv2.Rodrigues(rvec)[0]  # 结果是[[data]]，提出[data]
        camera_postion = -np.mat(rotM).T * np.mat(tvec)  # 得出相机相对现实坐标系的坐标
        print(camera_postion.T)
        theta_z = atan2(rotM[1][0], rotM[0][0])*57.2958
        theta_y = atan2(-rotM[2][0], sqrt(rotM[2][0] * rotM[2][0] + rotM[2][2] * rotM[2][2]))*57.2958
        theta_x = atan2(rotM[2][1], rotM[2][2])*57.2958
        print(theta_x, theta_y, theta_z)
        return 0

def Clens_up():
    pass
def Clens_down():
    pass
def Clens_left():
    pass
def Clens_right():
    pass

def get_target_speed():
    pass
speed = get_target_speed()
def PID_adjust(postion, angle, speed):
    """pid初始化
    目标位姿，速度
    动态调节"""


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            img = frame
            obj_2d_points = np.asarray(get_objct_2d_point(img))
            found, postion, angle = get_postion_and_angle(obj_2d_points)
            if found:
                PID_adjust(postion, angle, speed)




