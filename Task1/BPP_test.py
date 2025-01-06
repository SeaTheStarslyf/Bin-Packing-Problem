import gymnasium
from stable_baselines3 import PPO
from BPP_env import BinPackingEnv

# 创建环境
env = BinPackingEnv()  # 假设你已经实现了这个环境

model = PPO(
    "MultiInputPolicy",      # 输入策略：支持多种输入
    env,                     # 环境
    verbose=1,               # 输出训练日志
)
model.load('./ppo_binpacking_model.zip')

# 测试模型
obs, _info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

env.render()  # 如果有渲染方法的话
print(env.get_space_use_rate())
