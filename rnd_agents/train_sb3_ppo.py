#!/usr/bin/env python3
import os
import argparse
import yaml
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gym
from gym import spaces

# gym-floorplan imports
from gym_floorplan.envs.fenv_config import LaserWallConfig
from gym_floorplan.envs.master_env import SpaceLayoutGym


class FlattenMultiDiscrete(gym.ActionWrapper):
    """Wrap a MultiDiscrete action space as a single Discrete by flattening indices."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old = env.action_space
        if not isinstance(old, spaces.MultiDiscrete):
            raise ValueError("FlattenMultiDiscrete expects a MultiDiscrete action space")
        self.orig_nvec = old.nvec
        self.action_space = spaces.Discrete(int(np.prod(self.orig_nvec)))
        print(f"Flattened action space: {self.orig_nvec} -> Discrete({self.action_space.n})")

    def action(self, action):
        # incoming action is scalar idx -> map to vector action for the env
        vec = np.unravel_index(int(action), self.orig_nvec)
        return np.array(vec, dtype=np.int64)


def load_config(yaml_path: str = None, extra_hyper: dict = None):
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            hyper = yaml.safe_load(f) or {}
    else:
        hyper = {}
    if extra_hyper:
        hyper.update(extra_hyper)
    return hyper


def make_env_fn(hyper_params):
    def _init():
        cfg = LaserWallConfig(phase=hyper_params.get('phase', 'train'),
                              hyper_params=hyper_params).get_config()
        env = SpaceLayoutGym(cfg)
        # if env uses MultiDiscrete action, flatten to Discrete for SB3
        if isinstance(env.action_space, spaces.MultiDiscrete):
            env = FlattenMultiDiscrete(env)
        return env
    return _init


def pick_policy_for_env(env: gym.Env):
    obs_space = env.observation_space
    # if image-like Box (H,W,C) -> CnnPolicy; else MlpPolicy
    if isinstance(obs_space, spaces.Box) and len(obs_space.shape) >= 3:
        return "CnnPolicy"
    elif isinstance(obs_space, spaces.Dict):
        # check if any component is image-like
        for key, space in obs_space.spaces.items():
            if isinstance(space, spaces.Box) and len(space.shape) >= 3:
                return "MultiInputPolicy"
        return "MultiInputPolicy"
    return "MlpPolicy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--config-yaml", type=str, default="default_hps_env.yaml",
                        help="hyperparams YAML to pass to LaserWallConfig")
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--tensorboard-log", type=str, default="./tb_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    # Load config
    config_path = args.config_yaml if os.path.exists(args.config_yaml) else None
    hyper = load_config(config_path, extra_hyper={
        "phase": "train",
        "agent_name": "PPO",
        "n_rooms": 4,  # start small
        "plan_config_source_name": "fixed_test_config"
    })
    
    print(f"Using config: {hyper}")
    set_random_seed(args.seed)

    # make VecEnv
    env_fns = [make_env_fn(hyper) for _ in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Test environment
    print("Testing environment...")
    obs = vec_env.reset()
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

    # choose policy
    policy = pick_policy_for_env(vec_env.envs[0])
    print(f"Using policy: {policy}")

    # optional wandb
    if args.use_wandb:
        try:
            import wandb
            run = wandb.init(project="space_layout_rl", config={
                "timesteps": args.timesteps, 
                "seed": args.seed,
                "policy": policy,
                "hyper_params": hyper
            })
            tb_log_name = f"{args.tensorboard_log}/{run.id}"
        except Exception as e:
            print(f"Wandb init failed: {e}")
            tb_log_name = args.tensorboard_log
    else:
        tb_log_name = args.tensorboard_log

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="ppo_checkpoint"
    )

    # Create model
    model = PPO(policy, vec_env, verbose=1, tensorboard_log=tb_log_name,
                learning_rate=3e-4, n_steps=2048, batch_size=64)
    
    print("Starting training...")
    start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    elapsed = time.time() - start

    # Save final model
    save_path = os.path.join(args.save_dir, f"ppo_space_layout_final_{int(time.time())}")
    model.save(save_path)
    print(f"Training finished in {elapsed:.1f}s. Model saved to {save_path}")

    # Test trained model
    print("Testing trained model...")
    obs = vec_env.reset()
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(f"Step {i}: reward = {reward}, done = {done}")
        if done.any():
            break

    vec_env.close()


if __name__ == "__main__":
    main()