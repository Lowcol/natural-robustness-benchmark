# Evaluation script for trained PPO Atari agents
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.categorical import Categorical

from atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NaturalBackgroundWrapper,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = "eval_ppo_atari"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    
    # Evaluation specific arguments
    model_path: str = "runs/BreakoutNoFrameskip-v4__ppo_atari__1__1765048722/ppo_atari.cleanrl_model"
    """path to the trained model"""
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    num_episodes: int = 10
    """number of episodes to evaluate"""
    natural_video_folder: str = None
    """path to folder containing background videos for natural variant (None for standard env)"""
    deterministic: bool = False
    """if toggled, use deterministic actions (argmax) instead of sampling"""


def make_env(env_id, capture_video, run_name, natural_video_folder=None):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # Note: We don't use EpisodicLifeEnv during evaluation
        # so episodes run until actual game over
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # Note: We don't use ClipRewardEnv during evaluation
        # to get true game scores instead of clipped {-1, 0, 1}
        
        # Add natural background wrapper BEFORE resizing and grayscaling
        if natural_video_folder is not None:
            env = NaturalBackgroundWrapper(env, natural_video_folder)
        
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_deterministic_action(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        return logits.argmax(dim=-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Create run name with natural/standard indicator
    env_type = "natural" if args.natural_video_folder else "standard"
    run_name = f"{args.env_id}__{args.exp_name}__{env_type}__{args.seed}__{int(time.time())}"
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Environment: {args.env_id} ({env_type})")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    if args.natural_video_folder:
        print(f"Video folder: {args.natural_video_folder}")
    print("-" * 50)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup - use vectorized env with num_envs=1 for proper shape handling
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.capture_video, run_name, args.natural_video_folder)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    
    # Load the trained model
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()
    print(f"✓ Model loaded successfully from {args.model_path}")
    print("-" * 50)

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    episodes_completed = 0
    
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.Tensor(obs).to(device)
    
    while episodes_completed < args.num_episodes:
        with torch.no_grad():
            if args.deterministic:
                action = agent.get_deterministic_action(obs)
            else:
                action, _, _, _ = agent.get_action_and_value(obs)
        
        obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        obs = torch.Tensor(obs).to(device)
        
        # Check if episode finished
        if "final_info" in info:
            for final_info in info["final_info"]:
                if final_info and "episode" in final_info:
                    episode_reward = final_info["episode"]["r"]
                    episode_length = final_info["episode"]["l"]
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episodes_completed += 1
                    
                    print(f"Episode {episodes_completed}/{args.num_episodes}: "
                          f"Reward = {episode_reward:.2f}, Length = {episode_length}")
                    
                    if episodes_completed >= args.num_episodes:
                        break
    
    # Print summary statistics
    print("-" * 50)
    print("Evaluation Summary:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    
    envs.close()
    print(f"\n✓ Evaluation complete! Videos saved to videos/{run_name}/")
