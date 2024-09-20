import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
import yaml
import json
import sys
import copy
import pickle

sys.path.append("")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder, load_pickle, get_adjacency_matrix, scaled_Laplacian, load_adj
)

from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.STFDGCN import STFDGCN


def print_model(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    return param_count

@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # out_batch = model(x_batch)
        #
        # out_batch = SCALER.inverse_transform(out_batch)
        # loss = criterion(out_batch, y_batch)
        output = model(x_batch)

        y_pred = SCALER.inverse_transform(output)
        loss = criterion(y_pred, y_batch)

        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)

        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze() # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out #返回真实值和预测值 且为.cpu().numpy()


def train_one_epoch(model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None):
    global cfg, global_iter_count, global_target_length
    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)#x:(B, 12, N, 3)
        y_batch = y_batch.to(DEVICE)#y:(B, 12, N, 1)
        # out_batch = model(x_batch)
        # out_batch = SCALER.inverse_transform(out_batch)
        # loss = criterion(out_batch, y_batch)

        output = model(x_batch)

        y_pred = SCALER.inverse_transform(output)
        loss = criterion(y_pred, y_batch)

        batch_loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)#一个epoch需要进行669次 10700//B

    if scheduler:
        scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)#去cuda0

    wait = 0
    min_val_loss = np.inf#无穷

    train_loss_list = []#
    val_loss_list = []  #
    # ------------训练开始------------
    for epoch in range(max_epochs):#max_epochs=300

        train_loss = train_one_epoch(model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log)
        train_loss_list.append(train_loss)#1epoch train loss 加入到列表

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)#1epoch val loss 加入到列表

        if (epoch + 1) % verbose == 0:#[0,299]verbose=1 每个epcoh都打印
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,#1epoch train loss
                "Val Loss = %.5f" % val_loss,#1epoch val loss
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch#记录第几个epoch最后 0-299
            best_state_dict = copy.deepcopy(model.state_dict())#最后模型状态字典
        else:
            wait += 1
            if wait >= early_stop:#30poech不下降 停止训练
                break
    #------------训练结束------------

    model.load_state_dict(best_state_dict)
    #predict真实值和预测值  且为.cpu().numpy()
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))#用最好的模型预测train

    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))#用最好的模型预测val

    out_str = f"Early stopping at epoch: {epoch+1}\n"#停止训练时的epoch
    out_str += f"Best at epoch {best_epoch+1}:\n"#最好的第多少epoch
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]#最好epoch的train_loss值

    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % ( #最好时模型的train评估
        train_rmse,
        train_mae,
        train_mape,
    )

    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]#最好epoch的loss值

    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % ( #最好时模型的val评估
        val_rmse,
        val_mae,
        val_mape,
    )

    print_log(out_str, log=log)

    if plot:#画train和val loss下降图
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)#save=os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    return model #这个是训练时最好的模型


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    #输入训练时最好的模型，和test_dataloder
    y_true, y_pred = predict(model, testset_loader)

    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)#12时间步的评估

    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )

    out_steps = y_pred.shape[1]#12

    for i in range(out_steps):#进行每步的输出评估 Horizon1~12
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])

        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")

    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="PEMS08")

    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    #seed = torch.randint(1000, (1,))#set random seed here
    seed = 42
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    data_path = f"./data/{dataset}"
    data_path_csv = f"./data/{dataset}/distance.csv"

    model_name = STFDGCN.__name__

    with open(f"./config/{dataset}.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # ----------------------------------get adj_mx--------------------------------- #

    if dataset in ("PEMS04", "PEMS07", "PEMS08"):
        adj_mx = get_adjacency_matrix(data_path_csv, cfg.get("num_nodes"), 'connectivity', None)
    elif dataset == "METRLA":
        with open('./data/METRLA/adj_mx.pkl', 'rb') as file:
            adj_mx = pickle.load(file, encoding='latin1')[2]
    elif dataset == "PEMSBAY":
        with open('./data/PEMSBAY/adj_mx_bay.pkl', 'rb') as file:
            adj_mx = pickle.load(file, encoding='latin1')[2]
    else:
        #adj_mx = np.load('./data/PEMS03/adj.npy')
        with open('./data/PEMS03/adj_PEMS03.pkl', 'rb') as file:
            adj_mx = pickle.load(file)
    G = torch.from_numpy(adj_mx).type(torch.FloatTensor)
    adj_pfpb = [torch.tensor(i).to(DEVICE) for i in load_adj(G)]

    # -------------------------------- load model -------------------------------- #

    model = STFDGCN(adj_pfpb, **cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #
    print_log("---------", seed, "---------", log=log)
    print_log(dataset, log=log)

    (
        trainset_loader,#
        valset_loader,#
        testset_loader,#
        SCALER,#
    ) = get_dataloaders_from_index_data(
        data_path,#f"../data/{dataset}"
        tod=cfg.get("time_of_day"),#True
        dow=cfg.get("day_of_week"),#True
        batch_size=cfg.get("batch_size", 64),#
        log=log,
    )

    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #
    #loss
    if dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        #criterion = nn.HuberLoss()
        criterion = torch.nn.L1Loss()
        #criterion = MaskedMAELoss()
    else:
        criterion = MaskedMAELoss()


    #optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],#0.001
        weight_decay=cfg.get("weight_decay", 0),#0.0015
        eps=cfg.get("eps", 1e-8),#
    )

    # learning rate decay
    scheduler = None
    if cfg.get("lr_decay"):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["milestones"],#milestones: [25, 45, 65]
            gamma=cfg.get("lr_decay_rate", 0.1),#0.1
            verbose=False,
        )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)

    print_log(json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log)

    print_log(print_model(model), log=log)

    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)

    print_log(log=log)
    #得到的是训练时最好的模型
    model = train(
        model,#model = STAEformer(**cfg["model_args"])
        trainset_loader,#
        valset_loader,#
        optimizer,#Adam
        scheduler,#
        criterion,#nn.HuberLoss()
        clip_grad=cfg.get("clip_grad"),#clip_grad=cfg.get("clip_grad", 5)
        max_epochs=cfg.get("max_epochs", 200),#300
        early_stop=cfg.get("early_stop", 10),#30
        verbose=1,#1干嘛的
        log=log,#
        save=save,
    )
    #保存模型成功
    print_log(f"Saved Model: {save}", log=log)

    #用训练时表现最好的模型进行test
    test_model(model, testset_loader, log=log)
    log.close()
