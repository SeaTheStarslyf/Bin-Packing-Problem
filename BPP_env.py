import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Box import Box
from Box import generate_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import random

class BinPackingEnv(gym.Env):
    def __init__(self, container_size=(100, 100, 100), box_num_range=(10, 50)):
        super(BinPackingEnv, self).__init__()
        
        # 假设箱子的尺寸是 100x100x100
        self.container_size = container_size
        self.box_num_range = box_num_range
        self.L = container_size[0]
        self.W = container_size[1]
        self.H = container_size[2]
        
        # 假设物品的尺寸是随机的
        self.items = []
        self.height_map = np.zeros((self.L, self.W), dtype=np.int32)
        self.box_in_index = []
        self.steps = 0
        self.max_steps = 100
        self.volume_sum = 0
        self.max_height = 0
        
        # 状态空间：箱子的当前状态（height map, 物品，物品是否已经装载）
        self.observation_space = spaces.Dict({
            "items_state": spaces.Box(low=0, high=max(container_size), shape=(box_num_range[1], 6), dtype=np.int32),  # 每个盒子的属性 (l, w, h, x, y, z)
            "box_in_index_state": spaces.MultiBinary(box_num_range[1]),  # 每个盒子是否已经放入容器
            "height_map_state": spaces.Box(low=0, high=self.H, shape=(self.L, self.W), dtype=np.int32),  # 容器的高度图
        })
        
        # 动作空间：物品放置的位置以及旋转方式
        self.action_space = spaces.MultiDiscrete([
            self.L,  # position_x: x坐标的范围 [0, L)
            self.W,  # position_y: y坐标的范围 [0, W)
            box_num_range[1],  # box_index: 选择盒子的范围 [0, box_num_range[1])
            6  # rotation: 旋转方式的范围 [0, 6)
        ])

    def reset(self, seed=None, **kwargs):
        # 初始化箱子的状态
        box_num = np.random.randint(self.box_num_range[0], self.box_num_range[1] + 1)
        self.items = generate_boxes(self.container_size, box_num)
        self.height_map = np.zeros((self.L, self.W), dtype=np.int32)
        self.box_in_index = np.zeros(len(self.items), dtype=bool)
        self.volume_sum = 0
        self.steps = 0
        self.max_height = 0

        return self.get_state(), {}
    
    def step(self, action):
        x, y, box_idx, rotation = action
        self.steps += 1
        
        # 判断箱子索引是否合法
        if(box_idx >= len(self.items) or box_idx < 0):
            return self.get_state(), -1, self.done(), False, {"error": "invalid box index"}

        # 判断动作是否合法
        if (rotation < 0 or rotation >= 6):
            return self.get_state(), -1, self.done(), False, {"error": "invalid action"}

        # 判断是否越界
        if (x < 0 or x > self.L) or (y < 0 or y > self.W):
            return self.get_state(), -1, self.done(), False, {"error": "out of boundary"}
    
        # 判断箱子是否已经放置
        if self.box_in_index[box_idx]:
            return self.get_state(), -1, self.done(), False, {"error": "box already placed"}

        # 判断箱子是否可以放置
        if (self.valid_position(x, y, self.items[box_idx].get_rotate(rotation))):
            self.box_in_index[box_idx] = True
            self.items[box_idx].rotate(rotation)

            l, w, h = self.items[box_idx].dimensions
            max_height = np.max(self.height_map[x:x+l, y:y+w])
            self.max_height = max(self.max_height, max_height+h)
            num_max_height = np.sum(self.height_map[x:x+l, y:y+w] == max_height)
            self.height_map[x:x+l, y:y+w] = max_height + h
            self.items[box_idx].translate(x, y, max_height)
            self.volume_sum += self.items[box_idx].volume()
            return self.get_state(), self.get_reward(max_height+h, num_max_height/(l*w)), self.done(), False, {}
        else:
            return self.get_state(), -1, self.done(), False, {"error": "invalid placement"}

        # 根据动作更新环境状态（在这里只是一个简化版示例）
#        item = self.items[action]
#        reward = np.random.random()  # 这里可以根据装箱的效果设计奖励函数
#        done = np.random.choice([True, False])  # 终止条件可以根据实际情况设计
#        return self.state, reward, done, {}

    def valid_position(self, x, y, dimensions):
        l, w, h = dimensions
        if x + l > self.L or y + w > self.W:
            return False
        max_height = np.max(self.height_map[x:x+l, y:y+w])
        if max_height + h > self.H:
            return False
        return True
    
    def get_reward(self, max_height, solid_rate):
        use_rate = self.volume_sum / (self.L * self.W * self.max_height)
        height_penalty = 0.1 * (max_height / self.H)
        solid_penalty = (1 - solid_rate) * 0.3
        return use_rate - height_penalty - solid_penalty + 1
    
    def get_space_use_rate(self):
        return self.volume_sum / (self.L * self.W * self.H)
    
    def done(self):
        if self.steps >= self.max_steps or np.all(self.box_in_index):
            return True
        return False
    
    def get_state(self):
        items_state = np.zeros((self.box_num_range[1], 6), dtype=np.int32)
        box_in_index_state = np.ones(self.box_num_range[1], dtype=bool)
        items_state[:len(self.items)] = np.array([(box.dimensions + box.position) for box in self.items])
        box_in_index_state[:len(self.items)] = self.box_in_index
        return {
            "items_state": items_state,
            "box_in_index_state": box_in_index_state,
            "height_map_state": self.height_map
        }
    
    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 创建 X 和 Y 网格
        X = np.array([[0, self.L], [0, self.L]])
        Y = np.array([[0, 0], [self.W, self.W]])

        # 创建 Z 网格（平面高度，这里为0，表示底部）
        Z = np.array([[0, 0], [0, 0]])

        # 绘制容器的底面
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.1)

        # 为每个箱子生成一个随机颜色
        colors = [self.random_color() for _ in self.items]

        # 绘制每个箱子，颜色不同
        for box, color in zip(self.items, colors):
            if (self.box_in_index[box.id]):
                x, y, z = box.position
                l, w, h = box.dimensions

                # 绘制箱子为矩形
                ax.bar3d(x, y, z, l, w, h, color=color, alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.W)
        ax.set_zlim(0, self.H)
        plt.show()

    def random_color(self):
        # 随机生成一个RGB颜色值
        return (random.random(), random.random(), random.random())
        
