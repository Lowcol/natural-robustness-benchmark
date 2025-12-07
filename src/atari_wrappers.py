# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py)
# licensed under the MIT License.

from __future__ import annotations

import os
from typing import SupportsFloat, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]


class NaturalBackgroundWrapper(gym.ObservationWrapper):
    """
    Replaces the black background of the Atari game with a video from a folder.
    Must be applied to RGB environments (before Grayscale wrapper).
    """
    def __init__(self, env: gym.Env, video_folder: str) -> None:
        super().__init__(env)
        if cv2 is None:
            raise ImportError("opencv-python is required for NaturalBackgroundWrapper")
            
        self.video_folder = video_folder
        # Recursively search for video files in all subdirectories
        self.video_files = []
        for root, dirs, files in os.walk(video_folder):
            for f in files:
                if f.endswith(('.mp4', '.avi', '.mkv')) and not f.endswith(('.part', '.ytdl')):
                    self.video_files.append(os.path.join(root, f))
        
        if not self.video_files:
            raise RuntimeError(f"No videos found in {video_folder}!")
            
        print(f"Found {len(self.video_files)} videos in {video_folder}")
        self.cap = None
        
    def _load_random_video(self):
        if self.cap is not None:
            self.cap.release()
        
        video_path = np.random.choice(self.video_files)
        self.cap = cv2.VideoCapture(video_path)
        
    def reset(self, **kwargs):
        self._load_random_video()
        return super().reset(**kwargs)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # 1. Read next video frame
        ret, bg_frame = self.cap.read()
        
        # If video ends, loop it or pick new one
        if not ret:
            self._load_random_video()
            ret, bg_frame = self.cap.read()
            if not ret: # Fallback if video read fails
                return obs

        # 2. Resize video to match Game (H, W)
        h, w, _ = obs.shape
        bg_frame = cv2.resize(bg_frame, (w, h))

        # 3. Create Mask (Where is the game black?)
        # Atari Breakout background is pure black (0, 0, 0)
        # We use < 5 threshold to be safe against minor artifacts
        mask = (obs < 5).all(axis=2)

        # 4. Replace background
        obs[mask] = bg_frame[mask]
        
        return obs


class StickyActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sticky action.
    Paper: https://arxiv.org/abs/1709.06009
    """
    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int):
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    """
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.
    """
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        return np.sign(float(reward))


class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default).
    """
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        assert cv2 is not None, "OpenCV is not installed"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings
    """
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
        natural_video_folder: Optional[str] = None, # <--- NEW PARAMETER
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
            
        # --- NEW: Inject Natural Wrapper here (While env is still RGB) ---
        if natural_video_folder is not None:
            env = NaturalBackgroundWrapper(env, natural_video_folder)
        # -----------------------------------------------------------------
            
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)