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
        for i in range(self.length()-1):
            cur = self.head
            while cur is not None and cur.next is not None:
                after = cur.next
                if cur is self.head:
                    if cur.weight < after.weight:
                        self.head = after
                        cur.next = after.next
                        after.next = cur
                    else:
                        cur = cur.next
                else:
                    if cur.weight < after.weight:
                        before = self.head
                        while before.next is not cur:
                            before = before.next
                        before.next = after
                        cur.next = after.next
                        after.next = cur
                    else:
                        cur = cur.next

    def travel(self):
        cur = self.head
        while cur is not None:
            print(cur.weight, end=' ')
            cur = cur.next
        print('')



if __name__ is '__main__':
    ll = LinkedList()
    ll.append(1, (2, 3), 3)
    ll.append((2, 3), (2, 3), 3)
    ll.append((2, 3), (2, 3), 3)
    ll.append((2, 3), (2, 3), 3)
    ll.append((2, 3), (2, 3), 3)
    cur = ll.head
    i = 0
    while cur is not None:
        cur.weight += i
        i += 1
        cur = cur.next
    ll.sort()
    ll.travel()

