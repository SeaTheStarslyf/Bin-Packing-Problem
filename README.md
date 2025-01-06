# Bin-Packing-Problem
## 环境准备
Box.py：实现algorithm1，用于随机生成箱子

BPP_env.py：自定义的用于三维装箱问题的环境
## Task1

BPP_train.py：调用stable_baselines3中PPO进行强化学习训练

BPP_test.py：测试已有的本地模型

## Task2
MCTS.py：实现蒙特卡洛树，通过训练好的PPO模型指导路径的选择

MCTS_train.py：使用结合PPO模型的蒙特卡洛树进行装箱

## Task3
