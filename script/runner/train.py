import torch
import torch.nn as nn
import argparse
import os
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
from script.utils.infomanager import InfoManager
from script.utils.dataset import KoopmanDataset
from script.utils.network import ResidualNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Klinear_loss(state, action, net, mse_loss, traj_len, s_dim, gamma=0.99, alpha=0.1):
    Z_t = net.encode(state[:, 0, :])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(traj_len):
        Z_t_1 = net.forward(Z_t, action[:, i, :])
        Real_Z_t_1 = net.encode(state[:, i+1, :])

        beta *= gamma
        loss += beta * (mse_loss(Z_t_1, Real_Z_t_1)+alpha*mse_loss(Z_t_1[:, :s_dim], Z_t[:, :s_dim]))

        beta_sum += beta
        Z_t = Z_t_1

    return loss / beta_sum

def Keep_Distance_loss(state, action, net, traj_len, z_norm=2, x_norm=2):
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(traj_len):
        Z_t = net.encode(state[:, i, :])
        Z_t_1 = net.forward(Z_t, action[:, i, :])
        Z_distance = torch.norm(Z_t_1 - Z_t, p=z_norm)
        X_distance = torch.norm(state[:, i+1, :] - state[:, i, :], p=x_norm)
        loss += torch.norm(Z_distance - X_distance, p=1)
    return loss / traj_len

def Reg_loss(net):
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for param in net.lA.parameters():
        loss += torch.norm(param, p=2)
    for param in net.lB.parameters():
        loss += torch.norm(param, p=2)
    return loss

def Continuous_Predict_loss(state, action, net, mse_loss, traj_len, s_dim):
    Z_t = net.encode(state[:, 0, :])
    loss_x = torch.zeros(1,dtype=torch.float64).to(device)
    loss_z = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(traj_len):
        Z_t_1 = net.forward(Z_t, action[:, i, :])
        Real_Z_t_1 = net.encode(state[:, i+1, :])
        loss_x += mse_loss(Z_t_1[:, :s_dim], state[:, i+1, :])
        loss_z += mse_loss(Z_t_1, Real_Z_t_1)
        Z_t = Z_t_1
    return loss_x / traj_len, loss_z / traj_len

def Single_Predict_loss(state, action, net, mse_loss, traj_len, s_dim):
    loss_x = torch.zeros(1,dtype=torch.float64).to(device)
    loss_z = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(traj_len):
        Z_t = net.encode(state[:, i, :])
        Z_t_1 = net.forward(Z_t, action[:, i, :])
        Real_Z_t_1 = net.encode(state[:, i+1, :])
        loss_x += mse_loss(Z_t_1[:, :s_dim], state[:, i+1, :])
        loss_z += mse_loss(Z_t_1, Real_Z_t_1)
    return loss_x / traj_len, loss_z / traj_len

def Continuous_State_loss(state, action, net, traj_len, s_dim, dataset, robot):
    Z_t = net.encode(state[:, 0, :])
    if dataset.normalize:
        state = state * dataset.state_std + dataset.state_mean
    # loss_dof_pos, loss_dof_vel, loss_xy, loss_z, loss_orientation, loss_lin_vel, loss_ang_vel => 7 DoF
    loss = torch.zeros(7,dtype=torch.float64).to(device)
    for i in range(traj_len):
        Z_t_1 = net.forward(Z_t, action[:, i, :])
        X_t_1 = Z_t_1[:, :s_dim].clone()
        if dataset.normalize:
            X_t_1 = X_t_1 * dataset.state_std + dataset.state_mean
            X_t_1 = X_t_1.squeeze()
        if robot == "g1":
            loss[0] += torch.abs(X_t_1[:, 0:23] - state[:, i+1, 0:23]).mean()
            loss[1] += torch.abs(X_t_1[:, 23:46] - state[:, i+1, 23:46]).mean()
            loss[2] += 0
            loss[3] += torch.abs(X_t_1[:, 46:47] - state[:, i+1, 46:47]).mean()
            loss[4] += 0
            loss[5] += torch.abs(X_t_1[:, 47:50] - state[:, i+1, 47:50]).mean()
            loss[6] += torch.abs(X_t_1[:, 50:53] - state[:, i+1, 50:53]).mean()
        elif robot == "h1":
            loss[0] += torch.abs(X_t_1[:, 0:19] - state[:, i+1, 0:19]).mean()
            loss[1] += torch.abs(X_t_1[:, 19:38] - state[:, i+1, 19:38]).mean()
            loss[2] += 0
            loss[3] += torch.abs(X_t_1[:, 38:39] - state[:, i+1, 38:39]).mean()
            loss[4] += 0
            loss[5] += torch.abs(X_t_1[:, 39:42] - state[:, i+1, 39:42]).mean()
            loss[6] += torch.abs(X_t_1[:, 42:45] - state[:, i+1, 42:45]).mean()
        elif robot == "go2" or robot == "a1" or robot == "anymal-D":
            loss[0] += torch.abs(X_t_1[:, 0:12] - state[:, i+1, 0:12]).mean()
            loss[1] += torch.abs(X_t_1[:, 12:24] - state[:, i+1, 12:24]).mean()
            loss[2] += 0
            loss[3] += torch.abs(X_t_1[:, 24:25] - state[:, i+1, 24:25]).mean()
            loss[4] += torch.abs(X_t_1[:, 25:29] - state[:, i+1, 25:29]).mean()
            loss[5] += torch.abs(X_t_1[:, 29:32] - state[:, i+1, 29:32]).mean()
            loss[6] += torch.abs(X_t_1[:, 32:35] - state[:, i+1, 32:35]).mean()
        Z_t = Z_t_1
    return loss / traj_len
    
def trainer():
    task = args.task
    args.wandb_project = f"Koopman_for_{task}"
    if "anymal" in task:
        robot = "anymal-D"
    else:
        robot = task.split("_")[-2]
    print("[Task] ", task)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Robo_Dataset = KoopmanDataset(args.data_path, track_data_paths=args.track_data_paths, 
                                  data_num=args.data_num, track_data_num=args.track_data_num, random_start=args.random_start, 
                                  traj_len=args.traj_len, normalize=args.normalize, robot=robot)
    print("[Data] ", args.data_path)
    train_size = int(0.8 * len(Robo_Dataset))
    test_size = len(Robo_Dataset) - train_size 
    train_dataset, test_dataset = random_split(Robo_Dataset, [train_size, test_size])

    print("Data Size: ", Robo_Dataset.state_data.shape, Robo_Dataset.action_data.shape)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    s_dim = Robo_Dataset.obs_dim
    u_dim = Robo_Dataset.act_dim
    traj_len = Robo_Dataset.traj_len

    N_koopman = args.encode_dim + s_dim
    net = ResidualNetwork(N_koopman, u_dim, input_dim=s_dim, encode_dim=args.encode_dim, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device).double()
    print("[Network] Resnet")
    print(net)
    if args.resume:
        net.load_state_dict(torch.load(args.resume))
        print(f"[Load Model] {args.resume}")
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)

    iter_num = len(args.track_data_paths)
    Info_Manager = InfoManager(total_steps=args.epoch, args=args, iter_num=iter_num, task=task)
    Info_Manager.save_param(args, s_dim, u_dim, traj_len, N_koopman)
    Info_Manager.save_normalize_data(Robo_Dataset.state_mean, Robo_Dataset.state_std)

    for epoch in range(args.epoch):

        net.train()
        train_loss, train_K_loss, train_D_loss, train_R_loss = 0.0, 0.0, 0.0, 0.0
        for state, action in train_loader:
            K_loss = Klinear_loss(state, action, net, mse_loss, traj_len, s_dim, gamma=args.gamma)
            D_loss = Keep_Distance_loss(state, action, net, traj_len)
            R_loss = Reg_loss(net)
            loss = K_loss + args.keep_dis_loss * D_loss + args.reg_loss * R_loss
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            train_K_loss += K_loss.item()
            train_D_loss += D_loss.item()
            train_R_loss += R_loss.item()
        scheduler.step()

        train_loss /= (len(train_loader) / args.batch_size)
        train_K_loss /= (len(train_loader) / args.batch_size)
        train_D_loss /= (len(train_loader) / args.batch_size)
        train_R_loss /= (len(train_loader) / args.batch_size)

        net.eval()
        test_loss, test_K_loss, C_Pred_loss, S_Pred_loss, C_Z_Pred_loss, S_Z_Pred_loss, test_D_loss, test_R_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        state_loss = torch.zeros(7,dtype=torch.float64).to(device)
        with torch.no_grad():
            for state, action in test_loader:
                K_loss = Klinear_loss(state, action, net, mse_loss, traj_len, s_dim, gamma=args.gamma)
                C_Pred_loss, C_Z_Pred_loss = Continuous_Predict_loss(state, action, net, mse_loss, traj_len, s_dim)
                S_Pred_loss, S_Z_Pred_loss = Single_Predict_loss(state, action, net, mse_loss, traj_len, s_dim)
                D_loss = Keep_Distance_loss(state, action, net, traj_len)
                R_loss = Reg_loss(net)
                State_loss = Continuous_State_loss(state, action, net, traj_len, s_dim, Robo_Dataset, robot)

                loss = K_loss + args.keep_dis_loss * D_loss + args.reg_loss * R_loss

                test_loss += loss.item()
                test_K_loss += K_loss.item()
                C_Pred_loss += C_Pred_loss.item()
                S_Pred_loss += S_Pred_loss.item()
                C_Z_Pred_loss += C_Z_Pred_loss.item()
                S_Z_Pred_loss += S_Z_Pred_loss.item()
                test_D_loss += D_loss.item()
                test_R_loss += R_loss.item()
                state_loss += State_loss

            test_loss /= (len(test_loader) / args.batch_size)
            test_K_loss /= (len(test_loader) / args.batch_size)
            C_Pred_loss /= (len(test_loader) / args.batch_size)
            S_Pred_loss /= (len(test_loader) / args.batch_size)
            C_Z_Pred_loss /= (len(test_loader) / args.batch_size)
            S_Z_Pred_loss /= (len(test_loader) / args.batch_size)
            test_D_loss /= (len(test_loader) / args.batch_size)
            test_R_loss /= (len(test_loader) / args.batch_size)
            state_loss /= (len(test_loader) / args.batch_size)

        if epoch % args.save_interval == 0 or epoch == args.epoch-1:
            Info_Manager.save_model(net, f"model{epoch}.pt")
        if not args.debug:
            Info_Manager.add_scalar([("Train/Loss", train_loss, epoch), ("Train/Koopman Linear Loss", train_K_loss, epoch), 
                                    ("Train/Keep Distance Loss", train_D_loss, epoch), ("Train/Reg Loss", train_R_loss, epoch)])
            Info_Manager.add_scalar([("Test/Loss", test_loss, epoch), ("Test/Koopman Linear Loss", test_K_loss, epoch), 
                                    ("Test/Continuous Predict Loss", C_Pred_loss, epoch), 
                                    ("Test/Single Predict Loss", S_Pred_loss, epoch), ("Test/Keep Distance Loss", test_D_loss, epoch),
                                    ("Test/Continuous Z Predict Loss", C_Z_Pred_loss, epoch), ("Test/Single Z Predict Loss", S_Z_Pred_loss, epoch),
                                    ("Test/Reg Loss", test_R_loss, epoch), ("Test/DoF Pos Error", state_loss[0].item(), epoch),
                                    ("Test/DoF Vel Error", state_loss[1].item(), epoch), ("Test/Root Delta XY Error", state_loss[2].item(), epoch),
                                    ("Test/Root Z Error", state_loss[3].item(), epoch), ("Test/Root Orientation Error", state_loss[4].item(), epoch),
                                    ("Test/Root Linear Vel Error", state_loss[5].item(), epoch), ("Test/Root Angular Vel Error", state_loss[6].item(), epoch)])
        Info_Manager.update_pbar(f"Epoch {epoch+1}/{args.epoch} | Train Loss {train_loss} | Test Loss {test_loss}")
    
    if not args.debug:
        Info_Manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=["g1_flat", "h1_flat", "unitree_go2_flat", "unitree_a1_flat", "anymal_d_flat"])
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_num", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--encode_dim", type=int, default=169)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reg_loss',  type=float, default=0.0)
    parser.add_argument('--keep_dis_loss',  type=float, default=0.0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default="Koopman_for_legged_robot")
    parser.add_argument('--debug', action='store_false', default=True)
    parser.add_argument('--random_start', action='store_false', default=True)
    parser.add_argument('--normalize', action='store_false', default=True)
    parser.add_argument("--traj_len", type=int, default=-1, 
                        help="Notice that traj_len should be len(state trajectory) - 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--track_data_paths', nargs='+', type=str, default=[])
    parser.add_argument("--track_data_num", type=int, default=-1)
    args = parser.parse_args()
    trainer()
