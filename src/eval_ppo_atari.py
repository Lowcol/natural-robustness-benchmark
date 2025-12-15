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
import matplotlib.pyplot as plt

from atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    GaussianNoiseWrapper,
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    
    # Evaluation specific arguments
    model_path: str = "runs/BreakoutNoFrameskip-v4__ppo_atari__1__1765048722/ppo_atari.cleanrl_model"
    """path to the trained model"""
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    num_episodes: int = 30
    """number of episodes to evaluate"""
    natural_video_folder: str = "video-David"
    """path to folder containing background videos for natural variant (None for standard env)"""
    deterministic: bool = False
    """if toggled, use deterministic actions (argmax) instead of sampling"""
    gaussian_noise: bool = False
    """if toggled, add maximum Gaussian noise (std=50) to observations"""


def make_env(env_id, capture_video, run_name, natural_video_folder=None, gaussian_noise=False):
    def thunk():
        # Always create RGB env when we may capture or inject backgrounds
        env = gym.make(env_id, render_mode="rgb_array") if capture_video else gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # Note: We don't use EpisodicLifeEnv during evaluation
        # so episodes run until actual game over
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # Note: We don't use ClipRewardEnv during evaluation
        # to get true game scores instead of clipped {-1, 0, 1}

        # Add natural background before resizing/grayscale so it is visible in videos
        if natural_video_folder is not None:
            env = NaturalBackgroundWrapper(env, natural_video_folder)
        
        # Add Gaussian noise if enabled (max noise: std=50)
        if gaussian_noise:
            env = GaussianNoiseWrapper(env, std=50)

        # Ensure recorded video plays at the expected speed (60fps base / 4-frame skip = 15fps)
        if env.metadata.get("render_fps") in (None, 0):
            env.metadata["render_fps"] = 15

        # Capture video after the background is injected (keeps color in recordings)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

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
    if args.gaussian_noise:
        env_type += "_gaussian_noise"
    run_name = f"{args.env_id}__{args.exp_name}__{env_type}__{args.seed}__{int(time.time())}"
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Environment: {args.env_id} ({env_type})")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    if args.natural_video_folder:
        print(f"Video folder: {args.natural_video_folder}")
    if args.gaussian_noise:
        print(f"Gaussian noise: enabled (std=50)")
    print("-" * 50)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup - use vectorized env with num_envs=1 for proper shape handling
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.capture_video, run_name, args.natural_video_folder, args.gaussian_noise)]
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
                    # Convert to Python scalars to avoid numpy formatting issues
                    episode_reward = float(final_info["episode"]["r"])
                    episode_length = int(final_info["episode"]["l"])
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
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Range: [{mean_reward - std_reward:.2f}, {mean_reward + std_reward:.2f}]")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    
    # Plot reward visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Episode rewards over time
    axes[0].plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-', linewidth=2)
    axes[0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0].fill_between(range(1, len(episode_rewards) + 1), 
                         np.mean(episode_rewards) - np.std(episode_rewards),
                         np.mean(episode_rewards) + np.std(episode_rewards),
                         alpha=0.2, color='r', label=f'±1 Std Dev')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Reward per Episode')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Reward distribution histogram
    axes[1].hist(episode_rewards, bins=10, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(episode_rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1].axvline(x=np.median(episode_rewards), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(episode_rewards):.2f}')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reward Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = f"videos/{run_name}/evaluation_results.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Reward plot saved to {plot_path}")
    plt.show()
    
    envs.close()
    print(f"✓ Evaluation complete! Videos saved to videos/{run_name}/")
