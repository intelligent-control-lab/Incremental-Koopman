import torch
from torch.utils.data import Dataset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KoopmanDataset(Dataset):
    def __init__(self, data_path, track_data_paths=[], data_num=-1, track_data_num=-1, random_start=True, traj_len=-1, normalize=True, robot="g1"):
        self.data = np.load(data_path, allow_pickle=True)
        assert data_num <= self.data["state_data"].shape[1], "data_num should be less than the number of data"
        self._traj_len = traj_len
        self.normalize = normalize

        if data_num != -1:
            self.state_data = torch.DoubleTensor(self.data["state_data"]).to(device)[:, :data_num, :]
            self.action_data = torch.DoubleTensor(self.data["action_data"]).to(device)[:, :data_num, :]
        else:
            self.state_data = torch.DoubleTensor(self.data["state_data"]).to(device)
            self.action_data = torch.DoubleTensor(self.data["action_data"]).to(device)
        

        if not random_start and traj_len != -1:
            assert traj_len <= self.state_data.shape[0], "traj_len should be less than the max length of trajectory"
            self.state_data = self.state_data[:traj_len+1, :, :]
            self.action_data = self.action_data[:traj_len, :, :]
        elif random_start and traj_len != -1:
            assert traj_len <= self.state_data.shape[0], "traj_len should be less than the max length of trajectory"
            sampled_state = torch.empty((traj_len+1, self.state_data.shape[1], self.state_data.shape[2]), dtype=torch.float64).to(device)
            sampled_action = torch.empty((traj_len, self.action_data.shape[1], self.action_data.shape[2]), dtype=torch.float64).to(device)
            for i in range(self.state_data.shape[1]):
                start_idx = np.random.randint(0, self.state_data.shape[0]-traj_len-1)
                sampled_state[:, i, :] = self.state_data[start_idx:start_idx+traj_len+1, i, :]
                sampled_action[:, i, :] = self.action_data[start_idx:start_idx+traj_len, i, :]
            self.state_data = sampled_state
            self.action_data = sampled_action

        for track_data_path in track_data_paths:
            track_data = np.load(track_data_path, allow_pickle=True)
            if track_data_num == -1:
                track_state_data = torch.DoubleTensor(track_data["state_data"]).to(device)[:traj_len+1, :, :]
                track_action_data = torch.DoubleTensor(track_data["action_data"]).to(device)[:traj_len, :, :]
            else:
                track_state_data = torch.DoubleTensor(track_data["state_data"]).to(device)[:traj_len+1, :track_data_num, :]
                track_action_data = torch.DoubleTensor(track_data["action_data"]).to(device)[:traj_len, :track_data_num, :]
            self.state_data = torch.cat((self.state_data, track_state_data), dim=1)
            self.action_data = torch.cat((self.action_data, track_action_data), dim=1)
        
        if robot == "g1":                                                   # w/o hand, xy, ori
            self.state_data = torch.cat([self.state_data[..., :23], self.state_data[..., 37:60], self.state_data[..., 76:77], self.state_data[..., 81:]], axis=2)
            self.action_data = self.action_data[..., :23]
        elif robot == "h1":                                                 # w/o hand, xy, ori
            self.state_data = torch.cat([self.state_data[..., :38], self.state_data[..., 40:41], self.state_data[..., 45:]], axis=2)
            self.action_data = self.action_data
        elif robot == "go2" or robot == "a1" or robot == "anymal-D":        # w/o xy
            self.state_data = torch.cat([self.state_data[..., :24], self.state_data[..., 26:]],  axis=2)
            self.action_data = self.action_data
        else:
            assert False, "Invalid robot type."

        # Normalize the data
        if self.normalize:
            self.state_mean = self.state_data.mean(dim=(0,1), keepdim=True)
            self.state_std = self.state_data.std(dim=(0,1), keepdim=True)
            self.state_data = (self.state_data - self.state_mean) / self.state_std

    def __len__(self):
        return self.state_data.shape[1]

    def __getitem__(self, idx):
        return self.state_data[:,idx,:], self.action_data[:,idx,:]

    @property
    def obs_dim(self):
        return self.state_data.shape[2]

    @property
    def act_dim(self):
        return self.action_data.shape[2]
    
    @property
    def traj_len(self):
        return self.action_data.shape[0] if self._traj_len == -1 else self._traj_len