import gymnasium as gym
import maze_env

env = gym.make("MazeEnv-v0", render_mode="human")

obs, _ = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # Random action
    obs, reward, done, _, _ = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}")

env.close()

