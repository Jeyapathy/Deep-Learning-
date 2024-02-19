import os  
import gym  
from stable_baselines3 import PPO  
from stable_baselines3.common.vec_env import DummyVecEnv  
from stable_baselines3.common.evaluation import evaluate_policy  
  
def load_checkpoint(env, model_class, checkpoint_path):  
    model = model_class.load(checkpoint_path, env=env)  
    return model  
  
def record_video(env, model, video_length, video_folder, video_name):  
    env = gym.wrappers.Monitor(env, video_folder, force=True, video_callable=lambda episode_id: True)  
    obs = env.reset()  
    for _ in range(video_length):  
        action, _ = model.predict(obs, deterministic=True)  
        obs, _, done, _ = env.step(action)  
        if done:  
            obs = env.reset()  
    env.close()  
  
def main(game_name, model_class, checkpoint_prefix, num_checkpoints, video_length):  
    env = gym.make(game_name)  
    env = DummyVecEnv([lambda: env])  # Wrap the environment  
  
    # Create the video folder if it doesn't exist  
    video_folder = 'videos'  
    os.makedirs(video_folder, exist_ok=True)  
  
    for i in range(1, num_checkpoints + 1):  
        checkpoint_path = f'{checkpoint_prefix}_{i * 20000}_steps'  
        model = load_checkpoint(env, model_class, checkpoint_path)  
          
        # Record a video  
        video_name = f'video_checkpoint_{i}'  
        print(f'Recording video for checkpoint {i}')  
        record_video(env, model, video_length, video_folder, video_name)  
  
if __name__ == '__main__':  
    game_name = 'CartPole-v1'  # Replace with the game you trained on  
    model_class = PPO  # Replace with the algorithm you used  
    checkpoint_prefix = 'rl_model'  # Replace with your checkpoint prefix  
    num_checkpoints = 10  # Total number of checkpoints  
    video_length = 1000  # Number of timesteps for the video  
  
    main(game_name, model_class, checkpoint_prefix, num_checkpoints, video_length)  

