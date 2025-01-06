import gymnasium
from BPP_env import BinPackingEnv
from MCTS import MCTS

import gymnasium
from stable_baselines3 import PPO
from BPP_env import BinPackingEnv

# 创建环境
env = BinPackingEnv(container_size=(10, 10, 10), box_num_range=(10, 50))

# 创建PPO模型
model = PPO(
    "MultiInputPolicy",      # 输入策略：支持多种输入
    env,                     # 环境
    verbose=1,               # 输出训练日志
)

# `total_timesteps` 是模型训练的总时间步数
total_timesteps = 10000

# 训练模型并显示进度条
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# 初始化环境
env = BinPackingEnv(container_size=(10, 10, 10), box_num_range=(10, 50))
next_state, _ = env.reset()  # 获取初始状态

done = False
while not done:
    mcts = MCTS(env, 20, model)
    best_action = mcts.run()
    next_state, reward, done, _, _ = env.step(best_action)

env.render()
print(env.get_space_use_rate())