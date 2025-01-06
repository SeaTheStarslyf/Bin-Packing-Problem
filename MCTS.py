# MCTS.py
import random
import numpy as np


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # 当前的环境状态（字典）
        self.parent = parent  # 父节点
        self.action = action  # 当前选择的动作
        self.children = []  # 子节点
        self.visit_count = 0  # 访问次数
        self.reward = 0  # 累积奖励

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_possible_actions(self.state))

    def get_possible_actions(self, state):
        # 从字典中提取 L, W, items 信息
        items_state = state["items_state"]
        box_in_index_state = state["box_in_index_state"]
        height_map_state = state["height_map_state"]

        L = height_map_state.shape[0]
        W = height_map_state.shape[1]

        # 提取未放置的箱子
        items = []
        for i, in_index in enumerate(box_in_index_state):
            if not in_index:
                l, w, h, x, y, z = items_state[i]
                items.append((l, w, h, x, y, z, i))

        possible_actions = []
        for box_idx, (l, w, h, x, y, z, _) in enumerate(items):
            for x in range(L):
                for y in range(W):
                    for rotation in range(6):
                        if self.valid_position(x, y, l, w, h, rotation, height_map_state):
                            possible_actions.append((x, y, box_idx, rotation))
        return possible_actions

    def valid_position(self, x, y, l, w, h, rotation, height_map_state):
        if x + l > height_map_state.shape[0] or y + w > height_map_state.shape[1]:
            return False
        max_height = np.max(height_map_state[x:x + l, y:y + w])
        if max_height + h > height_map_state.shape[1]:  # 这里应该是 height_map_state.shape[2] 或 self.H
            return False
        return True

    def get_best_child(self):
        # 返回最优子节点（即具有最高奖励/访问次数比的节点）
        best_child = None
        best_value = float('-inf')
        for child in self.children:
            value = child.reward / (child.visit_count + 1)  # 简单的平均奖励
            if value > best_value:
                best_value = value
                best_child = child
        return best_child


class MCTS:
    def __init__(self, env, max_simulations=1000, model=None):
        self.env = env  # 环境实例
        self.max_simulations = max_simulations  # 最大模拟次数
        self.model = model

    def run(self):
        root_state = self.env.reset()[0]  # 初始化状态
        root_node = MCTSNode(root_state)  # 创建根节点

        for _ in range(self.max_simulations):
            node = self.select_node(root_node)  # 选择节点
            if not node.is_fully_expanded():
                self.expand_node(node)  # 扩展节点
            reward = self.simulate(node)  # 蒙特卡洛模拟
            self.backpropagate(node, reward)  # 回溯

        return self.get_best_action(root_node)

    def select_node(self, node):
        # 从根节点开始选择最优子节点，直到叶节点
        while node.is_fully_expanded():
            node = node.get_best_child()
        return node

    def expand_node(self, node):
        # 在未完全扩展的节点上扩展一个子节点
        possible_actions = node.get_possible_actions(node.state)
        for action in possible_actions:
            new_state, reward, done, _, _ = self.env.step(action)  # 执行一步动作
            child_node = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child_node)

    def simulate(self, node):
        # 蒙特卡洛模拟，执行随机决策直到游戏结束
        state = node.state
        total_reward = 0
        done = False

        while not done:
            action, _states = self.model.predict(state) #使用PPO进行动作选择
            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            state = next_state

        return total_reward

    def backpropagate(self, node, reward):
        # 将模拟结果回溯至父节点
        while node is not None:
            node.visit_count += 1
            node.reward += reward
            node = node.parent

    def get_best_action(self, root_node):
        # 获取根节点下最优的动作
        best_child = root_node.get_best_child()
        return best_child.action if best_child else None