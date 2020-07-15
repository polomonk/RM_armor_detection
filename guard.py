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


transform = transforms.Compose([
    transforms.Resize(150),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

rectangle_LinkedList = LinkedList()  # 创建一个链表存放最小矩形的信息
before_target_points = []  # 用来存放之前匹配的矩形中心点对
next_frame_coordinate = []  # 预测下一帧图像目标的坐标
hit_coordinate = []  # 预测下n帧图像目标的坐标，即要打击的位置
Track = 0  # 追踪标志，0丢失，1最优，2次优
sub_optimal_node = None
left_up_point, right_down_point = [], []
img = None


def distance(point1, point2):  # 测量两个点直接的距离
    assert len(point1) == 2 and len(point2) == 2
    dist = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return dist


def image_proprecessing():  # 提取图像中的颜色并2值化
    global img
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


def find_contours(img_binary):  # 找到所有轮廓的最小矩形
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓信息和轮廓的层次信息
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 把轮廓信息用最小矩形包裹
        # print(rect)   # point = rect[0]   side = rect[1]   angle = rect[2]
        if rect[1][0] * rect[1][1] >= 0:  # 面积太小的没有价值
            rectangle_LinkedList.append(np.int0(rect[0]), rect[1], rect[2])  # 把找到的轮廓存入链表
    # cur = rectangle_LinkedList.head
    # while cur is not None:
    #     box = cv2.boxPoints((cur.point, cur.side, cur.orig_angle))
    #     box = np.int0(box)  # 转换成整数操作
    #     cv2.drawContours(img, [box], -1, (0, 255, 0), 2)  # 画出其最小外接矩形
    #     cur = cur.next


def image_intercept(point1, point2):  # 把图像ROI区域切割
    global img
    w_min = min(point2[0], point1[0])
    h_min = min(point2[1], point1[1])
    w_max = max(point2[0], point1[0])
    h_max = max(point2[1], point1[1])
    return img[h_min - 100:h_max + 100, w_min - 200:w_max + 200]


def area_weight(std_area, comp_w, comp_h, error=20):  # 对面积进行限制
    if comp_h * (comp_w - error) < std_area < comp_h * (comp_w + error):
        return 1
    else:
        return 0


def deviation_weight(std, comp, error=0.3):  # 对高度比限制
    if error < 0 or error > 1:
        return 0
    elif (1 - error) < std.height / comp.height < 1 / (1 - error):
        weight = 1
    else:
        weight = -1

    dist = distance(std.point, comp.point)
    hv = (std.height + comp.height) / 2
    dh = abs(std.height - comp.height)
    value = (hv * 2.5) / (dist * (1 + np.sin(dh / hv * 3.14 / 2)))
    if (1 - error) < value < (1 + error):
        weight -= abs(1 - value) ** 1
    else:
        weight -= 1
    return weight


def angle_weight(std, var, error=30):  # 2个矩形的角度差限制
    if var - error < std < var + error:  # 角度差值不超过error
        probability = 1 - pow(abs(std - var) / error, 3)  # 用3次函数进行拟合，正态分布也行
        return probability
    else:
        return -1


def parallel_weight(k, b, point, error=50):  # 用一个矩形的角度和中心点做直线，计算对应节点的中心点到直线的距离
    dist = abs(k * point[0] - point[1] + b) / np.sqrt(1 + k ** 2)
    if dist < error:  # 误差不超过error个像素点
        probability = 1 - pow(dist / error, 3)
        return probability
    else:
        return -1


def continue_weight(centre_point, error=30):
    if Track == 0:
        return 0
    else:
        if Track is 1:
            dist = distance(centre_point, before_target_points[-1])
            # print('distance', dist)
        elif sub_optimal_node is not None and Track is 2:
            sub_target_point = np.mean([sub_optimal_node.match_node.point, sub_optimal_node.point], axis=0),
            dist = distance(sub_target_point, centre_point)
        else:
            return 0
        probability = 1 - pow(dist / error, 3)
        return probability


def prediction_weight(centre_point, error=20):
    deviation = distance(centre_point, next_frame_coordinate)
    if deviation > error:
        return 0
    else:
        probability = 1 - pow(deviation / error, 3)
        return probability


def judge_weight():  # 决策函数
    compare_obj = rectangle_LinkedList.head
    while compare_obj is not None:  # 2层循环把所有匹配情况都遍历一遍
        cur = rectangle_LinkedList.head
        k = np.tan(np.radians(compare_obj.angle))  # 把角度转变成k斜率
        b = compare_obj.point[1] - k * compare_obj.point[0]  # 把中心点带入求出b
        while cur is not None:
            if cur is not compare_obj:  # 不允许自己匹配自己
                weight = 0  # 权重
                if 160 < cur.angle or cur.angle < 20:  # 角度再一定的区间内才计算权重
                    weight += 1 * deviation_weight(compare_obj, cur, error=0.3)  # 2个图像高度比值差，效果比面积好
                    weight += 2 * angle_weight(compare_obj.angle, cur.angle, error=15)  # 角度差距
                    weight += 1.5 * parallel_weight(k, b, cur.point, error=30)  # 点到直线的距离
                    # 可以增加高和匹配点的距离的关系
                    if len(before_target_points) > 0:  # 有迹可循
                        cur_target_point = np.mean([cur.point, compare_obj.point], axis=0)
                        weight += 1 * prediction_weight(cur_target_point, error=abs(cur.height + compare_obj.height) // 4) # 预测下一帧点的位置与现在位置比较
                    if compare_obj.weight < weight:  # 更新权重
                        compare_obj.weight = weight
                        compare_obj.match_node = cur
            cur = cur.next
        compare_obj = compare_obj.next


def kalman_filter(points, prediction=3):
    next_frame_coordinate.clear()  # 清除之前的预测，开始新的预测
    hit_coordinate.clear()
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
        next_frame_coordinate.append((f_mat * x_mat).tolist()[0][0])  # f_mat只预测下一帧的坐标
        hit_coordinate.append((np.mat([[1, prediction], [0, 1]]) * x_mat).tolist()[0][0])  # 预测count帧后的坐标
    return next_frame_coordinate, hit_coordinate  # 返回预测下一帧的x, y 和 下count帧的x, y


def show_result(cur):  # 结果可视化
    global left_up_point, right_down_point
    if cur.match_node is not None:
        # print(cur.area, cur.match_node.area)
        box1 = np.int0(cv2.boxPoints((cur.point, cur.side, cur.orig_angle)).tolist())  # 把匹配矩形转化成矩形的坐标点
        box2 = np.int0(cv2.boxPoints((cur.match_node.point, cur.match_node.side, cur.match_node.orig_angle)).tolist())
        if box1[0][0] < box2[0][0]:  # 把坐标点转化成目标矩形的4个角点
            left_box = box1
            right_box = box2
        else:
            left_box = box2
            right_box = box1
        left_up_point = [min(left_box[0][0], left_box[3][0]), min(left_box[3][1], right_box[2][1])]
        right_down_point = [max(right_box[1][0], right_box[2][0]), max(left_box[0][1], right_box[1][1])]
        return left_up_point, right_down_point
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


def refresh_state(optimal_node):  # 更新状态
    global Track
    if optimal_node.match_node is not None:  # 找到了权重最大的一对点
        optim_target_point = np.mean([optimal_node.point, optimal_node.match_node.point], axis=0)   # 当前一对点的中心位置
        # last_point = before_target_points[-1]  # 前一对点的中心位置
        reference_error = abs((optimal_node.height + optimal_node.match_node.height) // 4 + abs(
            optimal_node.height - optimal_node.match_node.height))  # 参考误差
        if distance(next_frame_coordinate, optim_target_point) < reference_error:  # 如果当前位置和预测位置差距不大则认为找到了目标
            Track = 1
            # cv2.circle(img, tuple(np.int0(hit_coordinate)), 3, color=(122, 45, 200), thickness=3)  # 预测目标的位置
        else:
            if sub_optimal_node is not None and sub_optimal_node.match_node is not None:  # 次优点与现在的位置差距也较大则认为目标丢失
                sub_target_point = np.mean([sub_optimal_node.point, sub_optimal_node.match_node.point], axis=0)  # 次优节点的中心位置
                if distance(sub_target_point, optim_target_point) < reference_error:
                    Track = 2  # 继续追踪次优点
                    before_target_points.clear()  # 之前收集的点没有意义了
                    before_target_points.append(sub_optimal_node)  # 按顺序添加新的目标点
                    before_target_points.append(optimal_node)
            else:  # 目标丢失
                Track = 0
                before_target_points.clear()  # 之前收集的点没有意义了


def find_optimal_node(low_limit=4):
    global sub_optimal_node
    sub_optimal_node = None
    rectangle_LinkedList.sort_by_weight()  # 按权重从大到小排列desc
    head = rectangle_LinkedList.head
    if head is None or head.weight < low_limit:  # 如果开头就没东西就不用看后面了
        return None

    cur = rectangle_LinkedList.head
    while cur is not None and cur.weight > low_limit:  # 保留权值超过阈值的
        cur = cur.next
    if cur is not None:  # 欲练神功，
        cur.next = None
    rectangle_LinkedList.remove(cur)

    duplicate_matched_node = None  # 被多次匹配的节点
    cur = rectangle_LinkedList.head
    while cur is not None:  # 查找是否有被重复匹配的节点，以便后续操作
        after = cur.next
        while after is not None:
            if cur.match_node is after.match_node:
                duplicate_matched_node = cur.match_node
                break
            after = after.next
        cur = cur.next

    if duplicate_matched_node:  # 如果有重复匹配的情况
        if duplicate_matched_node is not head.match_node and duplicate_matched_node is not head:  # 权重最大的节点可以确定就定为最优节点
            optimal_node = head
            sub_optimal_node = duplicate_matched_node
        else:  # 把被重复匹配的节点保留，其余节点榨干剩余价值后删除
            cur = rectangle_LinkedList.head
            while cur is not None:
                if cur.match_node is duplicate_matched_node or cur is duplicate_matched_node:
                    pass
                else:
                    if sub_optimal_node is None:  # 不满足条件的都不要了
                        sub_optimal_node = cur
                    rectangle_LinkedList.remove(cur)
                cur = cur.next

            while rectangle_LinkedList.length() > 2:  # 去掉面积小的节点，留下2个面积最大的节点作为匹配节点
                min_area = rectangle_LinkedList.head.area()
                min_area_node = rectangle_LinkedList.head
                cur = rectangle_LinkedList.head
                while cur is not None:
                    if cur.area() < min_area:
                        min_area = cur.area()
                        min_area_node = cur
                    cur = cur.next
                rectangle_LinkedList.remove(min_area_node)

            head = rectangle_LinkedList.head  # 更新节点匹配情况
            head.match_node = head.next
            optimal_node = head
    else:
        optimal_node = head
        if head.next is not None and head.next is head.match_node:
            sub_optimal_node = head.next.next
        else:
            sub_optimal_node = head.next

    return optimal_node


def main():
    global img
    global next_frame_coordinate, hit_coordinate, before_target_points, sub_optimal_node
    global left_up_point, right_down_point
    rectangle_LinkedList.clear()  # 清空链表开始操作新的帧
    # img_cp = img.copy()  # 后续操作会破坏原图，先备份
    img_binary = image_proprecessing()  # 预处理，提取目标颜色，会改变源图
    find_contours(img_binary)  # 找到面积合适的最小矩形。用二值图找轮廓，画在原图上
    judge_weight()  # 进行权重判断
    optimal_node = find_optimal_node()

    if optimal_node is not None:  # 如果这帧图像拿到了最优节点
        if distance(optimal_node.point, optimal_node.match_node.point) is not 0:
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
            left_up_point = (min(left_box[0][0], left_box[1][0]), min(left_box[2][1], left_box[1][1], right_box[2][1]))
            right_down_point = (max(right_box[3][0], right_box[2][0]), max(left_box[0][1], right_box[0][1], right_box[3][1]))
        # 把最大权重的点放进历史目标点
        before_target_points.append(np.mean([optimal_node.point, optimal_node.match_node.point], axis=0).tolist())
        if len(before_target_points) > 5:  # 用来预测的点数量，这里只保留4个
            before_target_points.pop(0)

        next_frame_coordinate, hit_coordinate = kalman_filter(before_target_points, prediction=3)  # 预测下一帧图和下n张图的坐标
        refresh_state(optimal_node)  # 更新追踪的状态位
        if Track == 1:  # 追踪到了目标
            cv2.circle(img, tuple(np.int0(hit_coordinate)), 3, color=(122, 45, 200), thickness=3)  # 画出预测目标的位置
    else:
        before_target_points.clear()  # 这帧图像没有目标则清空之前保留的坐标点
    return img


def drew_rect(img):
    color = (0, 255, 0)
    cv2.line(img, (left_up_point[0], right_down_point[1]), left_up_point, color=color, thickness=2)
    cv2.line(img, (left_up_point[0], right_down_point[1]), right_down_point, color=color, thickness=2)
    cv2.line(img, (right_down_point[0], left_up_point[1]), left_up_point, color=color, thickness=2)
    cv2.line(img, (right_down_point[0], left_up_point[1]), right_down_point, color=color, thickness=2)


if __name__ == '__main__':
    globals()['img']
    import time
    print('loading model...')
    model = ConvNet()
    model.load_state_dict(torch.load('dec.pth'))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()
    print('model loading compeleted')
    apd = 10

    video = r'D:\Document\py\opencv\demo\red_video\text11.avi'
    cap = cv2.VideoCapture(video)
    ret, img = cap.read()
    while ret:
        st = time.time()

        img = main()
        torch.no_grad()
        dst = img[left_up_point[1]-apd: right_down_point[1]+apd, left_up_point[0]-apd: right_down_point[0]+apd]
        dst = cv2.resize(dst, (64, 64))
        dst = Image.fromarray(dst).convert("RGB")
        data = transform(dst).unsqueeze(0)
        out = model(data.to(DEVICE)).to("cpu").item()
        out = 0 if out < 0.5 else 1
        if out == 0:
            # cv2.rectangle(img, left_up_point, right_down_point, (0, 255, 0), 0)
            drew_rect(img)
        else:
            pass
        cv2.imshow('dst', img)
        cv2.waitKey(30)
        ret, img = cap.read()

        et = time.time()
        # print('time', et - st)
