import torch
import wandb
import time
import yaml
import os
from tqdm import tqdm
import numpy as np

from rsl_rl.utils.wandb_utils import WandbSummaryWriter

class InfoManager():
    def __init__(self, total_steps, args, iter_num, task="g1"):
        self.pbar = tqdm(total=total_steps)

        curr_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        dir_name = f"Iter{iter_num}_{curr_time}"
        self.logdir = os.path.join("./logs", task, dir_name)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        cfg = vars(args)
        if not args.debug:
            self.writer = WandbSummaryWriter(log_dir=self.logdir, flush_secs=10, cfg=cfg)
    
    def add_scalar(self, scalar_list):
        for key, value, step in scalar_list:
            self.writer.add_scalar(key, value, step)

    def update_pbar(self, info):
        self.pbar.set_description(info)
        self.pbar.update(1)
    
    def save_param(self, args, s_dim, u_dim, traj_len, N_koopman):
        params = vars(args)
        params["state_dim"] = s_dim
        params["action_dim"] = u_dim
        params["traj_len"] = traj_len
        params["N_koopman"] = N_koopman
        with open(os.path.join(self.logdir, "param.yaml"), "w") as f:
            yaml.dump(params, f)
    
    def save_model(self, model, name):
        self.model_path = os.path.join(self.logdir, 'checkpoint')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(model.state_dict(), os.path.join(self.model_path, name))
    
    def close(self):
        self.pbar.close()
        self.writer.save_file(self.model_path)
        self.writer.stop()
    
    def save_normalize_data(self, state_mean, state_std):
        data = {
            "state_mean": state_mean.cpu().numpy(),
            "state_std": state_std.cpu().numpy()
        }
        np.savez(os.path.join(self.logdir, "normalize_data.npz"), **data)