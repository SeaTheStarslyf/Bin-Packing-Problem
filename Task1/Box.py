import math
import numpy as np
from typing import Tuple, List

# 定义 Box 类
class Box:
    def __init__(self, length, width, height, x, y, z, id):
        # 长宽高三元组
        self.dimensions = (length, width, height)
        # 位置 (x, y, z)
        self.position = (x, y, z)
        # 旋转：使用四元数表示
 #       self.rotation = rotation
        # 编号
        self.id = id

    def __repr__(self):
        return (f"Box(id={self.id}, dimensions={self.dimensions}, "
                f"position={self.position}, rotation={self.rotation})")

    def volume(self):
        """计算箱子的体积"""
        length, width, height = self.dimensions
        return length * width * height

    def translate(self, dx, dy, dz):
        """平移箱子位置"""
        self.position = (self.position[0] + dx, self.position[1] + dy, self.position[2] + dz)

    def get_rotate(self, rotation):
        """获得旋转箱子后信息"""
        if rotation == 0:
            return (self.dimensions[0], self.dimensions[1], self.dimensions[2])
        if rotation == 1:
            return (self.dimensions[0], self.dimensions[2], self.dimensions[1])
        if rotation == 2:
            return (self.dimensions[1], self.dimensions[0], self.dimensions[2])
        if rotation == 3:
            return (self.dimensions[1], self.dimensions[2], self.dimensions[0])
        if rotation == 4:
            return (self.dimensions[2], self.dimensions[0], self.dimensions[1])
        if rotation == 5:
            return (self.dimensions[2], self.dimensions[1], self.dimensions[0])

    def rotate(self, rotation):
        """旋转箱子"""
        if rotation == 0:
            self.dimensions = (self.dimensions[0], self.dimensions[1], self.dimensions[2])
        if rotation == 1:
            self.dimensions = (self.dimensions[0], self.dimensions[2], self.dimensions[1])
        if rotation == 2:
            self.dimensions = (self.dimensions[1], self.dimensions[0], self.dimensions[2])
        if rotation == 3:
            self.dimensions = (self.dimensions[1], self.dimensions[2], self.dimensions[0])
        if rotation == 4:
            self.dimensions = (self.dimensions[2], self.dimensions[0], self.dimensions[1])
        if rotation == 5:
            self.dimensions = (self.dimensions[2], self.dimensions[1], self.dimensions[0])


def generate_boxes(container_size: Tuple[float, float, float] = (100, 100, 100), N: int = 10) -> List[Box]:
        id_counter = 0
        items = [Box(*container_size, 0, 0, 0, id_counter)]
        id_counter += 1
#        N = np.random.randint(n_min, n_max + 1)
        while len(items) < N:
            volumes = [item.volume() for item in items]
            vol_probs = np.array(volumes) / sum(volumes)
            selected_idx = np.random.choice(len(items), p=vol_probs)
            selected_box = items.pop(selected_idx)

            edge_lengths = selected_box.dimensions
            axis_probs = np.array(edge_lengths) / sum(edge_lengths)
            axis = np.random.choice(3, p=axis_probs)

            edge_length = edge_lengths[axis]
            min_split = max(1, int(0.1 * edge_length))
            max_split = min(edge_length - 1, int(0.9 * edge_length))

            if max_split <= min_split:
                items.append(selected_box)
                continue

            split_position = np.random.randint(min_split, max_split + 1)
            dimension = list(selected_box.dimensions)
            dimension[axis] = split_position
            box1 = Box(*dimension, *selected_box.position, selected_box.id)
            dimension[axis] = edge_length - split_position
            box2 = Box(*dimension, *selected_box.position, id_counter)
            id_counter += 1

            box1.rotate(np.random.randint(0, 6))
            box2.rotate(np.random.randint(0, 6))
            items.append(box1)
            items.append(box2)

        items.sort(key=lambda box: box.id)  # 根据id属性进行排序
        return items