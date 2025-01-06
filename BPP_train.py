import gymnasium
from stable_baselines3 import PPO
from BPP_env import BinPackingEnv

# 创建环境
env = BinPackingEnv()  # 假设你已经实现了这个环境

# 创建PPO模型
model = PPO(
    "MultiInputPolicy",      # 输入策略：支持多种输入
    env,                     # 环境
    verbose=1,               # 输出训练日志
)

# 使用 tqdm 包装训练过程，以显示训练进度条
# `total_timesteps` 是模型训练的总时间步数
total_timesteps = 10000

# 训练模型并显示进度条
model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
# 保存训练好的模型
model.save("ppo_binpacking_model")

# 测试模型
obs, _info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

env.render()  # 如果有渲染方法的话
print(env.get_space_use_rate())
