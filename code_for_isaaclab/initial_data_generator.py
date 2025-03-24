# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Initial RL data generator")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-Play-v0", help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)

parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="RL Policy update interval")
parser.add_argument("--max_episode_len", type=int, default=None, help="Max length of single simulation episode.")
parser.add_argument("--collect_data", action="store_true", default=False, help="Whether to collect data.")
parser.add_argument("--data_dir", type=str, default=None, help="Data save directory.")
parser.add_argument("--noise_scale", type=float, default=0.0, help="Used for generating reference repository. \
                                    Scale for uniform noise added to saved state. [-noise_scale, noise_scale)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video or args_cli.collect_data:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime
from tqdm import tqdm
import re

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # create isaac environment
    if args_cli.max_episode_len:
       env_cfg.episode_length_s = args_cli.max_episode_len * env_cfg.sim.dt * env_cfg.decimation
    if args_cli.collect_data:
        args_cli.num_steps_per_env = args_cli.max_episode_len if args_cli.max_episode_len else 1000
        args_cli.video_length = args_cli.num_steps_per_env
        args_cli.video_interval = args_cli.num_steps_per_env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # wrap for video recording
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"Viz-rlLog-{args_cli.load_run.split('/')[-1]}-Envnum{args_cli.num_envs}-Stepnum{args_cli.num_steps_per_env}-Iter{args_cli.max_iterations}"
    log_dir = os.path.join(log_root_path, log_dir)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    scene = env.unwrapped.scene

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    if args_cli.collect_data:
        data = {
            "state_data": [],
            "action_data": [],
            #"body_state": [],
        }

        state_data_buffer = [[] for _ in range(env.num_envs)]
        action_data_buffer = [[] for _ in range(env.num_envs)]

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    for it in tqdm(range(args_cli.max_iterations)):
        # run everything in inference mode
        with torch.inference_mode():
            for i in range(args_cli.num_steps_per_env):
                if args_cli.collect_data:
                    dof_pos = scene['robot'].data.joint_pos.cpu().numpy()
                    dof_vel = scene['robot'].data.joint_vel.cpu().numpy()
                    root_state = scene['robot'].data.root_state_w.cpu().numpy()
                    state = np.concatenate([dof_pos, dof_vel, root_state], axis=1)
                    for j in range(env.num_envs):
                        if len(state_data_buffer[j]) == 0:
                            state_data_buffer[j].append(state[j, :])

                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, _ = env.step(actions)
                
                if args_cli.collect_data:
                    dof_pos = scene['robot'].data.joint_pos.cpu().numpy()
                    dof_vel = scene['robot'].data.joint_vel.cpu().numpy()
                    root_state = scene['robot'].data.root_state_w.cpu().numpy()
                    state = np.concatenate([dof_pos, dof_vel, root_state], axis=1)
                    for j in range(env.num_envs):
                        state_data_buffer[j].append(state[j, :])
                        action_data_buffer[j].append(actions[j, :].cpu().numpy())
                        if dones[j] == 1:
                            if len(action_data_buffer[j]) < env.max_episode_length:
                                state_data_buffer[j] = []
                                action_data_buffer[j] = []
                            elif len(action_data_buffer[j]) == env.max_episode_length:
                                state_data_buffer[j] = state_data_buffer[j][:-1]
                                action_data_buffer[j] = action_data_buffer[j][:-1]
                                data["state_data"].append(np.array(state_data_buffer[j]))
                                data["action_data"].append(np.array(action_data_buffer[j]))
                                state_data_buffer[j] = []
                                action_data_buffer[j] = []
                            else:
                                assert False, "Episode length is greater than max_episode_length, something is wrong"
    
    if args_cli.collect_data:
        data["state_data"] = np.array(data["state_data"]).transpose(1, 0, 2)
        data["action_data"] = np.array(data["action_data"]).transpose(1, 0, 2)
        if args_cli.noise_scale > 0.0:
            data["state_data"] += (np.random.rand(*data["state_data"].shape)-0.5)*2.0*args_cli.noise_scale
        print(f"[INFO]: State Date Shape: {data['state_data'].shape}  Action Data Shape: {data['action_data'].shape}")

        data_dir = args_cli.data_dir if args_cli.data_dir else log_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        def extract_datetime_from_path(path):
            pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
            match = re.search(pattern, path)
            return match.group() if match else None
        np.savez(os.path.join(data_dir, f"{extract_datetime_from_path(args_cli.load_run)}_trajnum{data['action_data'].shape[1]}_trajlen{env.max_episode_length}.npz"), **data)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
