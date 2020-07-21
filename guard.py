"""
    编码格式:utf-8
    guard.py
"""
from my_bag import *
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


class ImageGeometryDetection():
    def __init__(self):
        self.img = None
        self.Rect = RectangleLinkedList()  # 创建一个链表存放最小矩形的信息
        self.before_target_points = []  # 用来存放之前匹配的矩形中心点对
        self.next_frame_coordinate = []  # 预测下一帧图像目标的坐标
        self.hit_coordinate = []  # 预测下n帧图像目标的坐标，即要打击的位置
        self.Track = 0  # 追踪标志，0丢失，1最优，2次优
        self.sub_optimal_node = None
        self.left_up_point, self.right_down_point = [], []

    def distance(self, point1, point2):  # 测量两个点直接的距离
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def image_proprecessing(self):  # 提取图像中的颜色并2值化
        lower_bgr = np.array([150, 100, 150])  # bgr很亮部分的上下限
        upper_bgr = np.array([220, 220, 255])
        mask = cv2.inRange(img, lowerb=lower_bgr, upperb=upper_bgr)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  # 平滑图像
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # 限制亮度

        # binary = cv2.GaussianBlur(binary, (3, 3), 0)  # 平滑图像
        # _, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)  # 限制亮度

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 开操作，消除噪点
        binary = cv2.erode(binary, kernel)
        binary = cv2.dilate(binary, kernel)
        return binary

    def find_contours(self, img_binary):  # 找到所有轮廓的最小矩形
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓信息和轮廓的层次信息
        for contour in contours:
            rect = cv2.minAreaRect(contour)  # 把轮廓信息用最小矩形包裹
            # print(rect)   # point = rect[0]   side = rect[1]   angle = rect[2]
            if rect[1][0] * rect[1][1] >= 0:  # 面积太小的没有价值
                self.Rect.append(np.int0(rect[0]), rect[1], rect[2])  # 把找到的轮廓存入链表
        # cur = Rect.head
        # while cur is not None:
        #     box = cv2.boxPoints((cur.point, cur.side, cur.orig_angle))
        #     box = np.int0(box)  # 转换成整数操作
        #     cv2.drawContours(img, [box], -1, (0, 255, 0), 2)  # 画出其最小外接矩形
        #     cur = cur.next

    def image_intercept(self, point1, point2):  # 把图像ROI区域切割
        global img
        w_min = min(point2[0], point1[0])
        h_min = min(point2[1], point1[1])
        w_max = max(point2[0], point1[0])
        h_max = max(point2[1], point1[1])
        return img[h_min - 100:h_max + 100, w_min - 200:w_max + 200]

    def area_weight(self, std_area, comp_w, comp_h, error=20):  # 对面积进行限制
        if comp_h * (comp_w - error) < std_area < comp_h * (comp_w + error):
            return 1
        else:
            return 0

    def deviation_weight(self, std, comp, error=0.3):  # 对高度比限制
        if error < 0 or error > 1:
            return 0
        elif (1 - error) < std.height / comp.height < 1 / (1 - error):
            weight = 1
        else:
            weight = -1

        dist = self.distance(std.point, comp.point)
        hv = (std.height + comp.height) / 2
        dh = abs(std.height - comp.height)
        value = (hv * 2.5) / (dist * (1 + np.sin(dh / hv * 3.14 / 2)))
        if (1 - error) < value < (1 + error):
            weight -= abs(1 - value) ** 1
        else:
            weight -= 1
        return weight

    def angle_weight(self, std, var, error=30):  # 2个矩形的角度差限制
        if var - error < std < var + error:  # 角度差值不超过error
            probability = 1 - pow(abs(std - var) / error, 3)  # 用3次函数进行拟合，正态分布也行
            return probability
        else:
            return -1

    def parallel_weight(self, k, b, point, error=50):  # 用一个矩形的角度和中心点做直线，计算对应节点的中心点到直线的距离
        dist = abs(k * point[0] - point[1] + b) / np.sqrt(1 + k ** 2)
        if dist < error:  # 误差不超过error个像素点
            probability = 1 - pow(dist / error, 3)
            return probability
        else:
            return -1

    def continue_weight(self, centre_point, error=30):
        if self.Track == 0:
            return 0
        else:
            if self.Track is 1:
                dist = self.distance(centre_point, self.before_target_points[-1])
                # print('distance', dist)
            elif self.sub_optimal_node is not None and self.Track is 2:
                sub_target_point = np.mean([self.sub_optimal_node.match_node.point, self.sub_optimal_node.point],
                                           axis=0),
                dist = self.distance(sub_target_point, centre_point)
            else:
                return 0
            probability = 1 - pow(dist / error, 3)
            return probability

    def prediction_weight(self, centre_point, error=20):
        deviation = self.distance(centre_point, self.next_frame_coordinate)
        if deviation > error:
            return 0
        else:
            probability = 1 - pow(deviation / error, 3)
            return probability

    def judge_weight(self):  # 决策函数
        compare_obj = self.Rect.head
        while compare_obj is not None:  # 2层循环把所有匹配情况都遍历一遍
            cur = self.Rect.head
            k = np.tan(np.radians(compare_obj.angle))  # 把角度转变成k斜率
            b = compare_obj.point[1] - k * compare_obj.point[0]  # 把中心点带入求出b
            while cur is not None:
                if cur is not compare_obj:  # 不允许自己匹配自己
                    weight = 0  # 权重
                    if 160 < cur.angle or cur.angle < 20:  # 角度再一定的区间内才计算权重
                        weight += 1 * self.deviation_weight(compare_obj, cur, error=0.3)  # 2个图像高度比值差，效果比面积好
                        weight += 2 * self.angle_weight(compare_obj.angle, cur.angle, error=15)  # 角度差距
                        weight += 1.5 * self.parallel_weight(k, b, cur.point, error=30)  # 点到直线的距离
                        # 可以增加高和匹配点的距离的关系
                        if len(self.before_target_points) > 0:  # 有迹可循
                            cur_target_point = np.mean([cur.point, compare_obj.point], axis=0)
                            weight += 1 * self.prediction_weight(cur_target_point, error=abs(
                                cur.height + compare_obj.height) // 4)  # 预测下一帧点的位置与现在位置比较
                        if compare_obj.weight < weight:  # 更新权重
                            compare_obj.weight = weight
                            compare_obj.match_node = cur
                cur = cur.next
            compare_obj = compare_obj.next

    def kalman_filter(self, points, prediction=3):
        self.next_frame_coordinate.clear()  # 清除之前的预测，开始新的预测
        self.hit_coordinate.clear()
        z_mat = np.mat(points)  # 定义x的初始状态
        for i in [0, 1]:  # 对x, y分别进行预测
            x_mat = np.mat([[points[0][i], 0], [0, 0]])  # 定义初始状态协方差矩阵
            p_mat = np.mat(np.eye(2))  # 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
            f_mat = np.mat([[1, 1], [0, 1]])  # 定义观测矩阵
            h_mat = np.mat([1, 0])  # 定义观测噪声协方差
            r_mat = np.mat([0.1])  # 定义状态转移矩阵噪声
            for j in range(len(points)):
                x_predict = f_mat * x_mat
                p_predict = f_mat * p_mat * f_mat.T
                kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
                x_mat = x_predict + kalman * (z_mat[j, i] - h_mat * x_predict)
                p_mat = (np.eye(2) - kalman * h_mat) * p_predict
            self.next_frame_coordinate.append((f_mat * x_mat).tolist()[0][0])  # f_mat只预测下一帧的坐标
            self.hit_coordinate.append((np.mat([[1, prediction], [0, 1]]) * x_mat).tolist()[0][0])  # 预测count帧后的坐标
        return self.next_frame_coordinate, self.hit_coordinate  # 返回预测下一帧的x, y 和 下count帧的x, y

    def show_result(self, target_node):  # 结果可视化
        if target_node.match_node is not None:
            # print(target_node.area, target_node.match_node.area)
            box1 = np.int0(
                cv2.boxPoints((target_node.point, target_node.side, target_node.orig_angle)).tolist())  # 把匹配矩形转化成矩形的坐标点
            box2 = np.int0(cv2.boxPoints((target_node.match_node.point, target_node.match_node.side,
                                          target_node.match_node.orig_angle)).tolist())
            if box1[0][0] < box2[0][0]:  # 把坐标点转化成目标矩形的4个角点
                left_box = box1
                right_box = box2
            else:
                left_box = box2
                right_box = box1
            self.left_up_point = [min(left_box[0][0], left_box[3][0]), min(left_box[3][1], right_box[2][1])]
            self.right_down_point = [max(right_box[1][0], right_box[2][0]), max(left_box[0][1], right_box[1][1])]
            # if distance(left_box[0], left_box[1]) > distance(left_box[0], left_box[3]):
            #     left_down_point = tuple(left_box[0])
            #     left_up_point = tuple(left_box[1])
            # else:
            #     left_down_point = tuple(left_box[1])
            #     left_up_point = tuple(left_box[2])
            # if distance(right_box[0], right_box[1]) > distance(right_box[0], right_box[3]):
            #     right_down_point = tuple(right_box[3])
            #     right_up_point = tuple(right_box[2])
            # else:
            #     right_down_point = tuple(right_box[0])
            #     right_up_point = tuple(right_box[3])
            # color = (0, 255, 0)
            # cv2.line(img, left_down_point, left_up_point, color=color, thickness=2)
            # cv2.line(img, left_down_point, right_down_point, color=color, thickness=2)
            # cv2.line(img, right_up_point, left_up_point, color=color, thickness=2)
            # cv2.line(img, right_up_point, right_down_point, color=color, thickness=2)

    def refresh_state(self, optimal_node):  # 更新状态
        if optimal_node.match_node is not None:  # 找到了权重最大的一对点
            optim_target_point = np.mean([optimal_node.point, optimal_node.match_node.point], axis=0)  # 当前一对点的中心位置
            # last_point = before_target_points[-1]  # 前一对点的中心位置
            reference_error = abs((optimal_node.height + optimal_node.match_node.height) // 4 + abs(
                optimal_node.height - optimal_node.match_node.height))  # 参考误差
            if self.distance(self.next_frame_coordinate,
                             optim_target_point) < reference_error:  # 如果当前位置和预测位置差距不大则认为找到了目标
                self.Track = 1
                # cv2.circle(img, tuple(np.int0(self.hit_coordinate)), 3, color=(122, 45, 200), thickness=3)  # 预测目标的位置
            else:
                if self.sub_optimal_node is not None and self.sub_optimal_node.match_node is not None:      # 次优点与现在的位置差距也较大则认为目标丢失
                    sub_target_point = np.mean([self.sub_optimal_node.point, self.sub_optimal_node.match_node.point],
                                               axis=0)  # 次优节点的中心位置
                    if self.distance(sub_target_point, optim_target_point) < reference_error:
                        self.Track = 2  # 继续追踪次优点
                        self.before_target_points.clear()  # 之前收集的点没有意义了
                        self.before_target_points.append(self.sub_optimal_node)  # 按顺序添加新的目标点
                        self.before_target_points.append(optimal_node)
                else:  # 目标丢失
                    self.Track = 0
                    self.before_target_points.clear()  # 之前收集的点没有意义了

    def find_optimal_node(self, low_limit=4):
        self.sub_optimal_node = None
        self.Rect.sort_by_weight()  # 按权重从大到小排列desc
        head = self.Rect.head
        if head is None or head.weight < low_limit:  # 如果开头就没东西就不用看后面了
            return None

        cur = self.Rect.head
        while cur is not None and cur.weight > low_limit:  # 保留权值超过阈值的
            cur = cur.next
        if cur is not None:  # 欲练神功，
            cur.next = None
        self.Rect.remove(cur)

        duplicate_matched_node = None  # 被多次匹配的节点
        cur = self.Rect.head
        while cur is not None:  # 查找是否有被重复匹配的节点，以便后续操作
            after = cur.next
            while after is not None:
                if cur.match_node is after.match_node:
                    duplicate_matched_node = cur.match_node
                    break
                after = after.next
            cur = cur.next

        if duplicate_matched_node:  # 如果有重复匹配的情况
            if duplicate_matched_node is not head.match_node and duplicate_matched_node is not head:    # 权重最大的节点可以确定就定为最优节点
                optimal_node = head
                self.sub_optimal_node = duplicate_matched_node
            else:  # 把被重复匹配的节点保留，其余节点榨干剩余价值后删除
                cur = self.Rect.head
                while cur is not None:
                    if cur.match_node is duplicate_matched_node or cur is duplicate_matched_node:
                        pass
                    else:
                        if self.sub_optimal_node is None:  # 不满足条件的都不要了
                            self.sub_optimal_node = cur
                        self.Rect.remove(cur)
                    cur = cur.next

                while self.Rect.length() > 2:  # 去掉面积小的节点，留下2个面积最大的节点作为匹配节点
                    min_area = self.Rect.head.area()
                    min_area_node = self.Rect.head
                    cur = self.Rect.head
                    while cur is not None:
                        if cur.area() < min_area:
                            min_area = cur.area()
                            min_area_node = cur
                        cur = cur.next
                    self.Rect.remove(min_area_node)

                head = self.Rect.head  # 更新节点匹配情况
                head.match_node = head.next
                optimal_node = head
        else:
            optimal_node = head
            if head.next is not None and head.next is head.match_node:
                self.sub_optimal_node = head.next.next
            else:
                self.sub_optimal_node = head.next
        return optimal_node

    def drew_rect(self):
        color = (0, 255, 0)
        cv2.line(self.img, (self.left_up_point[0], self.right_down_point[1]), self.left_up_point, color=color,
                 thickness=2)
        cv2.line(self.img, (self.left_up_point[0], self.right_down_point[1]), self.right_down_point, color=color,
                 thickness=2)
        cv2.line(self.img, (self.right_down_point[0], self.left_up_point[1]), self.left_up_point, color=color,
                 thickness=2)
        cv2.line(self.img, (self.right_down_point[0], self.left_up_point[1]), self.right_down_point, color=color,
                 thickness=2)


class LoadNet():
    def __init__(self):
        class ConvNet(nn.Module):
            def __init__(self):
                super(ConvNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3)
                self.max_pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.max_pool2 = nn.MaxPool2d(2)
                self.conv3 = nn.Conv2d(64, 128, 3)
                self.max_pool3 = nn.MaxPool2d(2)
                self.conv4 = nn.Conv2d(128, 128, 3)
                self.max_pool4 = nn.MaxPool2d(2)
                self.fc1 = nn.Linear(6272, 512)
                self.fc2 = nn.Linear(512, 1)
                self.dropout = nn.Dropout(p=0.5)

            def forward(self, x):
                in_size = x.size(0)
                x = self.conv1(x)
                x = F.relu(x)
                x = self.max_pool1(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = self.max_pool2(x)
                x = self.conv3(x)
                x = F.relu(x)
                x = self.max_pool3(x)
                x = self.conv4(x)
                x = F.relu(x)
                x = self.max_pool4(x)
                # 展开
                x = x.view(in_size, -1)
                x = self.fc1(x)
                x = self.dropout(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = torch.sigmoid(x)
                return x

        self.transform = transforms.Compose([
            transforms.Resize(150),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.model = ConvNet()
        self.model.load_state_dict(torch.load('dec.pth'))
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.DEVICE)
        self.model.eval()

    def image_verification(self, image):
        image = cv2.resize(image, (150, 150))
        image = Image.fromarray(image).convert("RGB")
        data = self.transform(image).unsqueeze(0)
        out = self.model(data.to(self.DEVICE)).to("cpu").item()
        return out



def main():
    ImgDect.Rect.clear()  # 清空链表开始操作新的帧
    # img_cp = img.copy()  # 后续操作会破坏原图，先备份
    img_binary = ImgDect.image_proprecessing()  # 预处理，提取目标颜色，会改变源图
    ImgDect.find_contours(img_binary)  # 找到面积合适的最小矩形。用二值图找轮廓，画在原图上
    ImgDect.judge_weight()  # 进行权重判断
    optimal_node = ImgDect.find_optimal_node()

    if optimal_node is not None:  # 如果这帧图像拿到了最优节点
        if ImgDect.distance(optimal_node.point, optimal_node.match_node.point) is not 0:
            # show_result(optimal_node)  # 展示结果
            # 把匹配矩形转化成矩形的坐标点
            box1 = np.int0(cv2.boxPoints((optimal_node.point, optimal_node.side, optimal_node.orig_angle)).tolist())
            box2 = np.int0(cv2.boxPoints((optimal_node.match_node.point, optimal_node.match_node.side,
                                          optimal_node.match_node.orig_angle)).tolist())
            if box1[0][0] < box2[0][0]:  # 把坐标点转化成目标矩形的4个角点
                left_box = box1
                right_box = box2
            else:
                left_box = box2
                right_box = box1
            ImgDect.left_up_point = (
            min(left_box[0][0], left_box[1][0]), min(left_box[2][1], left_box[1][1], right_box[2][1]))
            ImgDect.right_down_point = (
            max(right_box[3][0], right_box[2][0]), max(left_box[0][1], right_box[0][1], right_box[3][1]))
        # 把最大权重的点放进历史目标点
        ImgDect.before_target_points.append(
            np.mean([optimal_node.point, optimal_node.match_node.point], axis=0).tolist())
        if len(ImgDect.before_target_points) > 5:  # 用来预测的点数量，这里只保留4个
            ImgDect.before_target_points.pop(0)

        ImgDect.next_frame_coordinate, ImgDect.hit_coordinate = ImgDect.kalman_filter(ImgDect.before_target_points,
                                                                                      prediction=3)  # 预测下一帧图和下n张图的坐标
        ImgDect.refresh_state(optimal_node)  # 更新追踪的状态位
        if ImgDect.Track == 1:  # 追踪到了目标
            cv2.circle(ImgDect.img, tuple(np.int0(ImgDect.hit_coordinate)), 3, color=(122, 45, 200),
                       thickness=3)  # 画出预测目标的位置
    else:
        ImgDect.before_target_points.clear()  # 这帧图像没有目标则清空之前保留的坐标点


if __name__ == '__main__':
    import time

    ImgDect = ImageGeometryDetection()

    print('loading model...')
    Net = LoadNet()
    print('model loading compeleted')

    apd = 10
    video = r'D:\Document\py\opencv\demo\red_video\text11.avi'
    cap = cv2.VideoCapture(video)
    ret, img = cap.read()
    while ret:
        st = time.time()
        ImgDect.img = img
        main()
        torch.no_grad()
        area2redetect = 0
        dst = ImgDect.img[max(ImgDect.left_up_point[1] - apd, 0): min(ImgDect.right_down_point[1] + apd, img.shape[0]),
              max(ImgDect.left_up_point[0] - apd, 0): min(ImgDect.right_down_point[0] + apd, img.shape[1])]

        out = Net.image_verification(dst)
        out = 0 if out < 0.5 else 1
        if out == 0:
            # cv2.rectangle(img, left_up_point, right_down_point, (0, 255, 0), 0)
            ImgDect.drew_rect()
        else:
            pass
        cv2.imshow('dst', img)
        cv2.waitKey(30)
        ret, img = cap.read()

        et = time.time()
        # print('time', et - st)
