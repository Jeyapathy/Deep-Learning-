import os  
import gym  
import argparse  
from stable_baselines3 import PPO  
  
# Function to create the game environment and wrap it for video recording  
def create_env(game_name, video_folder):  
    env = gym.make(game_name)  
    env = gym.wrappers.Monitor(env, video_folder, force=True)  
    return env  
  
# Function to load the trained model  
def load_policy(model_path, env):  
    model = PPO.load(model_path, env=env)  
    return model  
  
# Function to play the game using the loaded policy  
def play_game(env, model, num_episodes=1):  
    for episode in range(num_episodes):  
        obs = env.reset()  
        done = False  
        while not done:  
            action, _states = model.predict(obs, deterministic=True)  
            obs, reward, done, info = env.step(action)  
            env.render()  
    env.close()  
  
def main(model_path):  
    # Define the game and the directory for video output  
    game_name = 'CartPole-v1'  
    video_folder = 'videos'  
  
    # Create the environment and wrap it for video recording  
    env = create_env(game_name, video_folder)  
  
    # Load the trained policy/model  
    model = load_policy(model_path, env)  
  
    # Play the game using the loaded policy  
    play_game(env, model)  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Deploy a trained policy to play a game and record a video.')  
    parser.add_argument('model_path', type=str, help='Path to the trained policy/model/checkpoint to use.')  
    args = parser.parse_args()  
      
    # Ensure that the video folder exists  
    if not os.path.exists('videos'):  
        os.makedirs('videos')  
  
    # Run the main function with the provided model path  
    main(args.model_path)  