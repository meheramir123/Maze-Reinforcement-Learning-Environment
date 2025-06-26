from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import maze_env 


# Load environment 
env = gym.make("MazeEnv-v0")

# Check if the environment follows Gym API
check_env(env)

#  Create PPO model 
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./maze_tensorboard/"
)

#  Train 
print("Starting training...")
model.learn(total_timesteps=20000)
print("Training complete!")

#  Save 
model.save("ppo_maze_agent")
print("Model saved!")

#  Evaluate 
obs, _ = env.reset()
done = False

while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print(f"Position: {obs}, Reward: {reward}")

env.close()
