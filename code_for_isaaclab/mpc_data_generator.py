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
parser = argparse.ArgumentParser(description="MPC data generator")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=None, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=None, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-Play-v0", help="Name of the task.")

parser.add_argument("--start_idx", type=int, default=20, help="Start step for MPC to avoid initial floating situation")
parser.add_argument("--num_steps_per_env", type=int, default=200, help="RL Policy update interval")
parser.add_argument("--ref", type=str, help="Reference respository")
parser.add_argument("--data_dir", type=str, default=None, help="Data save directory.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
import yaml
import re

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedEnv, DirectMARLEnv, multi_agent_to_single_agent

device = torch.device("cuda:0" if torch.cuda.is_available() and not args_cli.cpu else "cpu")

"""Define Reference Trajectory and reset functions as follow."""
ref_data: dict
x_ref: torch.Tensor
shift_step: int = 3
num_envs = int(args_cli.ref.split("trajnum")[1].split("_")[0])
shift_buf = torch.zeros(num_envs).int().to(device) - shift_step
robot: str

def reset_root_state_specific(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset the asset root state to a specific position and velocity from the given tensor.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    global ref_data, shift_buf, shift_step
    ref_num = ref_data["state_data"].shape[1]

    shift_buf[env_ids] += shift_step

    root_states = ref_data["state_data"][shift_buf, torch.arange(shift_buf.shape[0])%ref_num][env_ids, -13:]

    positions = torch.cat([env.scene.env_origins[env_ids][:, :2], root_states[:, 2:3]], dim=-1)
    orientations = root_states[:, 3:7]
    velocities = root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1).float(), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities.float(), env_ids=env_ids)

def reset_joints_specific(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset the robot joints by the given tensor.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    global ref_data, shift_buf
    ref_num = ref_data["state_data"].shape[1]

    joint_states = ref_data["state_data"][shift_buf, torch.arange(shift_buf.shape[0])%ref_num][env_ids, :-13]

    joint_pos = joint_states[:, :joint_states.shape[1]//2]
    joint_vel = joint_states[:, joint_states.shape[1]//2:]

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos.float(), joint_vel.float(), env_ids=env_ids)

"""Define Koopman dynamics as follow."""
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, encode_dim)
        self.residual_connection = nn.Linear(input_dim, encode_dim) if input_dim != encode_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        return self.relu(out)

class ResidualNetwork(nn.Module): 
    def __init__(self, Nkoopman, u_dim, input_dim, encode_dim, hidden_dim=256, num_blocks=3):
        super(ResidualNetwork, self).__init__()
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.final_fc = nn.Linear(hidden_dim, encode_dim)

        self.Nkoopman = Nkoopman
        self.u_dim = u_dim 
        self.s_dim = input_dim

        self.lA = nn.Linear(Nkoopman,Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim,Nkoopman,bias=False)

    def encode_only(self, x):
        x = self.initial_fc(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_fc(x)
        return x

    def encode(self, x):
        return torch.cat([x,self.encode_only(x)],axis=-1)
    
    def forward(self, z, u):
        return self.lA(z)+self.lB(u)

"""MPC Design"""
def Torch_MPC(net, x_k, X_ref, H, m, n, device, robot):
    z_k = net.encode(x_k).permute(1, 0)
    Z_ref = net.encode(X_ref).permute(1, 0, 2).reshape(-1, (H+1)*n)

    A = net.lA.weight.data
    B = net.lB.weight.data
    Q = torch.eye(n).to(device).double()
    R = torch.eye(m).to(device).double() * 0.0
    F = torch.eye(n).to(device).double()

    Q[net.s_dim:, net.s_dim:] *= 0.0
    F[net.s_dim:, net.s_dim:] *= 0.0

    M = torch.cat([torch.matrix_power(A, i) for i in range(H+1)], dim=0).to(device).double()
    C = torch.zeros((H+1)*n, H*m)
    for i in range(1, H+1):
        for j in range(H):
            if j <= i - 1:
                C[i*n:(i+1)*n, j*m:(j+1)*m] = torch.matrix_power(A, i-j-1) @ B
    C = C.to(device).double()

    Q_hat = torch.block_diag(*([Q]*H + [F]))
    R_hat = torch.block_diag(*([R]*H))

    p = 2 * (R_hat + C.T @ Q_hat @ C)
    q = 2 * (z_k.T @ M.T - Z_ref) @ Q_hat @ C

    U_k = (-0.5 * torch.inverse(0.5*p) @ q.T).reshape(H*m, -1)
    u_k = U_k[:m, :].permute(1, 0)

    res = (M@z_k+C@U_k-Z_ref.T).T@Q_hat@(M@z_k+C@U_k-Z_ref.T)
    res = res[torch.arange(res.shape[0]), torch.arange(res.shape[0])]

    return u_k, res

def main():
    global ref_data, x_ref, shift_buf, num_envs

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    robot = ""
    if "g1" in agent_cfg.experiment_name:
        robot = "g1"
    elif "h1" in agent_cfg.experiment_name:
        robot = "h1"
    elif "go2" in agent_cfg.experiment_name:
        robot = "go2"
    elif "a1" in agent_cfg.experiment_name:
        robot = "a1"
    elif "anymal_d" in agent_cfg.experiment_name:
        robot = "anymal-D"
    else:
        assert False, "Robot not supported."
    
    # shut down the randomization of robot pose and velocity
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }

    # define event terms for resetting the robot
    env_cfg.events.reset_base = EventTerm(
        func=reset_root_state_specific,
        mode="reset",
        params={},
    )
    env_cfg.events.reset_robot_joints = EventTerm(
        func=reset_joints_specific,
        mode="reset",
        params={},
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    """Define Koopman dynamics as follow."""
    resume_path = os.path.join(args_cli.load_run, "checkpoint")
    last_checkpoint = "model"+str(max([int(i.split("model")[1].split(".pt")[0]) for i in os.listdir(resume_path)]))+".pt"
    if args_cli.checkpoint:
        resume_path = os.path.join(resume_path, args_cli.checkpoint)
    else:
        resume_path = os.path.join(resume_path, last_checkpoint)
    
    param_path = resume_path.split("checkpoint")[0]+"param.yaml"
    with open(param_path, "r") as f:
        params = yaml.safe_load(f)
    
    encode_dim = params["encode_dim"]
    N_koopman = params["N_koopman"]
    u_dim = params["action_dim"]
    s_dim = params["state_dim"]
    traj_len = params["traj_len"]
    normalize = params["normalize"]

    net = ResidualNetwork(N_koopman, u_dim, s_dim, encode_dim, hidden_dim=256, num_blocks=3).to(device).double()
    net.load_state_dict(torch.load(resume_path))
    net.eval()

    # load reference trajectory and global variables
    ref_data = np.load(args_cli.ref, allow_pickle=True)
    state_data = torch.DoubleTensor(ref_data["state_data"]).to(device)[args_cli.start_idx:, ...]
    action_data = torch.DoubleTensor(ref_data["action_data"]).to(device)[args_cli.start_idx:, ...]
    ref_data = OrderedDict({"state_data": state_data, "action_data": action_data})
    x_ref = state_data.clone()
    u_ref = action_data.clone()
    if "g1" in robot:
        x_ref = torch.cat([x_ref[..., :23], x_ref[..., 37:60], x_ref[..., 76:77], x_ref[..., 81:]], axis=-1)
        u_ref = u_ref[..., :23]
    elif "h1" in robot:
        x_ref = torch.cat([x_ref[..., :38], x_ref[..., 40:41], x_ref[..., 45:]], axis=-1)
        u_ref = u_ref
    elif "go2" in robot or "a1" in robot or "anymal-D" in robot:
        x_ref = torch.cat([x_ref[..., :24], x_ref[..., 26:]], axis=-1)
        u_ref = u_ref
    else:
        assert False, "Robot not supported."
    action_l  = u_ref.min(dim=0).values
    action_u  = u_ref.max(dim=0).values

    # load normalization data and normalize x_ref
    state_mean, state_std = None, None
    if normalize:
        normalize_data_path = resume_path.split("checkpoint")[0] + "normalize_data.npz"
        normalize_data = np.load(normalize_data_path)
        state_mean = torch.tensor(normalize_data["state_mean"], dtype=torch.double).to(device)
        state_std = torch.tensor(normalize_data["state_std"], dtype=torch.double).to(device)
        x_ref = ((x_ref-state_mean)/state_std).squeeze(0)

    # create isaac environment
    env_cfg.episode_length_s = (x_ref.shape[0] - traj_len - 1) * env_cfg.sim.dt * env_cfg.decimation
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    args_cli.video_length = x_ref.shape[0] - traj_len - 1 if not args_cli.video_length else args_cli.video_length
    args_cli.video_interval = x_ref.shape[0] - traj_len - 1 if not args_cli.video_interval else args_cli.video_interval
    video_dir = f"Viz-KoopmanLog-{args_cli.load_run.split('/')[-1]}-Ref-{args_cli.ref.split('/')[-1].split('.')[0]}"
    video_dir = os.path.join(log_root_path, video_dir)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(video_dir, "videos"),
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

    def get_obs():
        dof_pos = scene['robot'].data.joint_pos
        dof_vel = scene['robot'].data.joint_vel
        root_state = scene['robot'].data.root_state_w
        states = torch.cat([dof_pos, dof_vel, root_state], axis=-1)
        if "g1" in robot:
            states = torch.cat([states[..., :23], states[..., 37:60], states[..., 76:77], states[..., 81:]], axis=-1)
        elif "h1" in robot:
            states = torch.cat([states[..., :38], states[..., 40:41], states[..., 45:]], axis=-1)
        elif "go2" in robot or "a1" in robot or "anymal-D" in robot:
            states = torch.cat([states[..., :24], states[..., 26:]], axis=-1) 
        if normalize:
            states = ((states-state_mean)/state_std).squeeze(0)
        return states 
    
    data = {
        "state_data": [],
        "action_data": []
    }
    state_data_buffer = [[] for _ in range(env.num_envs)]
    action_data_buffer = [[] for _ in range(env.num_envs)]
    
    max_len = x_ref.shape[0] - traj_len - 1 - 10 # 10 is reserved for shift_step <= 10
    num_steps = args_cli.num_steps_per_env if args_cli.num_steps_per_env <= max_len else max_len
    ref_num = x_ref.shape[1]
    data_num = 30000

    obs = get_obs()
    with torch.inference_mode():
        for i in tqdm(range(num_steps)):
            dof_pos = scene['robot'].data.joint_pos
            dof_vel = scene['robot'].data.joint_vel
            root_state = scene['robot'].data.root_state_w
            state = torch.cat([dof_pos, dof_vel, root_state], axis=1)
            for j in range(env.num_envs):
                if env.episode_length_buf[j] == 0:
                    # x, y don't follow ref_data
                    assert torch.abs(state[j, :-13] - ref_data["state_data"][shift_buf[j], j%ref_num, :-13]).mean() == 0.0, f"State mismatch: {j}, {shift_buf[j]}"
                    assert torch.abs(state[j, -11:] - ref_data["state_data"][shift_buf[j], j%ref_num, -11:]).mean() == 0.0, f"State mismatch: {j}, {shift_buf[j]}"
                if len(state_data_buffer[j]) == 0:
                    state_data_buffer[j].append(state[j, :])

            range_matrix = (env.episode_length_buf+shift_buf).unsqueeze(0) + torch.arange(traj_len+1).unsqueeze(1).to(device)
            sliced_x_ref = x_ref[range_matrix, (torch.arange(env.episode_length_buf.shape[0])%ref_num).unsqueeze(0)]

            actions, _ = Torch_MPC(net, obs, sliced_x_ref, traj_len, u_dim, N_koopman, device, robot)
            actions = actions.clamp(action_l, action_u)
            actions = torch.cat([actions, torch.zeros(env.num_envs, ref_data["action_data"].shape[-1] - u_ref.shape[-1]).to(device)], axis=-1)
            _, _, dones, _ = env.step(actions)
            obs = get_obs()

            dof_pos = scene['robot'].data.joint_pos
            dof_vel = scene['robot'].data.joint_vel
            root_state = scene['robot'].data.root_state_w
            state = torch.cat([dof_pos, dof_vel, root_state], axis=1)
            for j in range(env.num_envs):
                state_data_buffer[j].append(state[j, :])
                action_data_buffer[j].append(actions[j, :])
                if dones[j] == 1 or i == num_steps-1:
                    state_data_buffer[j] = state_data_buffer[j][:-1]
                    action_data_buffer[j] = action_data_buffer[j][:-1]
                    if len(action_data_buffer[j]) > 0:
                        data["state_data"].append(torch.stack(state_data_buffer[j], dim=0))
                        data["action_data"].append(torch.stack(action_data_buffer[j], dim=0))
                    state_data_buffer[j] = []
                    action_data_buffer[j] = []

    # check data shape
    remove_list = []
    for i in range(len(data["state_data"])):
        assert data["state_data"][i].shape[0] == data["action_data"][i].shape[0]+1, f"[Error]Incorrect data shape: State Data Shape: {data['state_data'][i].shape}  Action Data Shape: {data['action_data'][i].shape}"
        if data["state_data"][i].shape[0] <= traj_len + 1:
            remove_list.append(i)
    for i in remove_list[::-1]:
        data["state_data"].pop(i)
        data["action_data"].pop(i)

    # save data
    new_data = {
        "state_data": [],
        "action_data": [],
    }
    for i in tqdm(range(data_num)):
        random_idx = np.random.randint(0, len(data["state_data"]))
        random_start = np.random.randint(0, data["state_data"][random_idx].shape[0] - traj_len - 1)
        new_data["state_data"].append(data["state_data"][random_idx][random_start:random_start+traj_len+1, :])
        new_data["action_data"].append(data["action_data"][random_idx][random_start:random_start+traj_len, :])

    new_data["state_data"] = torch.stack(new_data["state_data"], dim=0).transpose(0, 1).cpu().numpy()
    new_data["action_data"] = torch.stack(new_data["action_data"], dim=0).transpose(0, 1).cpu().numpy()
    print(f"[INFO]: State Date Shape: {new_data['state_data'].shape}  Action Data Shape: {new_data['action_data'].shape}")
    if not os.path.exists(args_cli.data_dir):
        os.makedirs(args_cli.data_dir)

    def extract_datetime_from_path(path):
        pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
        match = re.search(pattern, path)
        return match.group() if match else None
    np.savez(os.path.join(args_cli.data_dir, f"{extract_datetime_from_path(args_cli.load_run)}_trajnum{new_data['action_data'].shape[1]}_trajlen{new_data['action_data'].shape[0]}.npz"), **new_data)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
