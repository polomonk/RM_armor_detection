"""
    编码格式:utf-8
    my_bag.py
"""


class Node(object):
    def __init__(self, point, side, angle):
        self.point = point      # 中心点的位置
        self.side = side        # 包含长和宽的数组，宽为离水平线最近的边
        self.orig_angle = angle
        if side[0] >= side[1]:      # 这里的角度是负值，因为图像的坐标系在左上角
            self.angle = angle+90
            self.height = side[0]
            self.width = side[1]
        else:
            self.angle = angle      # 宽到水平线的角度
            self.width = side[0]
            self.height = side[1]
        self.weight = 0
        self.match_node = None
        self.next = None

    def area(self):
        return self.side[0] * self.side[1]


class RectangleLinkedList(object):
    def __init__(self, node=None):
        self.head = node

    def is_empty(self):
        return self.head is None

    def clear(self):
        self.head = None

    def length(self):
        cur = self.head
        length = 0
        while cur is not None:
            cur = cur.next
            length += 1
        return length

    def remove(self, node):
        cur = self.head
        pre = None
        while cur is not None:
            if cur is node:
                if cur is self.head:
                    self.head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

    def append(self, point, side, angle):
        node = Node(point, side, angle)
        cur = self.head
        if self.head is None:
            self.head = node
        else:
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def index2item(self, index):
        cur = self.head
        count = 0
        while count is not index:
            cur = cur.next
            count += 1
        return cur

    def sort_by_weight(self):
        def swap(node1, node2):
            if node1 is node2:
                return
            pre1 = self.head
            per2 = self.head
            cur = self.head
            while cur.next is not None:
                if cur.next is node1:
                    pre1 = cur
                elif cur.next is node2:
                    pre2 = cur
                cur = cur.next
            if node1 is self.head:
                if node1.next is node2:
                    node1.next = node2.next
                    node2.next = node1
                    self.head = node2
                else:
                    nxt1 = node1.next
                    pre2.next = node1
                    node1.next = node2.next
                    node2.next = nxt1
                    self.head = node2
            elif node2 is self.head:
                if node2.next is node1:
                    node2.next = node1.next
                    node1.next = node2
                    self.head = node1
                else:
                    nxt2 = node2.next
                    pre1.next = node2
                    node2.next = node1.next
                    node1.next = nxt2
                    self.head = node1
            else:
                if node1.next is node2:
                    pre1.next = node2
                    node1.next = node2.next
                    node2.next = node1
                elif node2.next is node1:
                    pre2.next = node1
                    node2.next = node1.next
                    node1.next = node2
                else:
                    nxt1 = node1.next
                    pre1.next = node2
                    pre2.next = node1
                    node1.next = node2.next
                    node2.next = nxt1

        def sort_step(bgn, end):
            if bgn is end or bgn.next is end:
                return bgn

            key = bgn.weight
            up = bgn  # pup指针的移动比pdw快
            dw = bgn
            while up is not end:
                if up.weight > key:
                    dw = dw.next
                    swap(up, dw)
                    up, dw = dw, up
                up = up.next
            swap(bgn, dw)
            bgn, dw = dw, bgn
            return bgn, dw

        def sort(bgn, end):
            if bgn is end or bgn.next is end:
                return
            bgn, mid = sort_step(bgn, end)
            sort(bgn, mid)
            sort(mid.next, end)

        bgn = self.head
        end = self.head
        while end is not None:
            end = end.next
        sort(bgn, end)
        # for i in range(self.length()-1):
        #     cur = self.head
        #     while cur is not None and cur.next is not None:
        #         after = cur.next
        #         if cur is self.head:
        #             if cur.weight < after.weight:
        #                 self.head = after
        #                 cur.next = after.next
        #                 after.next = cur
        #             else:
        #                 cur = cur.next
        #         else:
        #             if cur.weight < after.weight:
        #                 before = self.head
        #                 while before.next is not cur:
        #                     before = before.next
        #                 before.next = after
        #                 cur.next = after.next
        #                 after.next = cur
        #             else:
        #                 cur = cur.next

    def travel(self):
        cur = self.head
        while cur is not None:
            print(cur.weight, end=' ')
            cur = cur.next
        print('')



if __name__ is '__main__':
    pass
