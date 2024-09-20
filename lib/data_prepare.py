import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)


def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)#(17856,170,3) [133,0,0] 只有流量 后面是0

    features = [0]#[0, 1, 2]
    if tod:  #True
        features.append(1)
    if dow:  #True
        features.append(2)
    # if dom:
    #     features.append(3)        #D2STGNN 和 DDGCRN 都用此数据集
    data = data[..., features]#(17856,170,3) [flow,0~1,0-6]这个数据集用的直接处理好后的 D和W

    index = np.load(os.path.join(data_dir, "index.npz"))#['train', 'val', 'test']

    train_index = index["train"] #0:[0,10699]     1:[12,10711]    2:[24,10723]   #10700
    val_index = index["val"]     #0:[10700,14266] 1:[10712,14278] 2:[10724,14290]#3567
    test_index = index["test"]   #0:[14267,17832] 1:[14279,17844] 2:[14291,17856]#3566

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]#(10700,12,170,3)
    y_train = data[y_train_index][..., :1]#(10700,12,170,1)
    x_val = data[x_val_index]#(3567,12,170,3)
    y_val = data[y_val_index][..., :1]#(3567,12,170,1)
    x_test = data[x_test_index]#(3566,12,170,3)
    y_test = data[y_test_index][..., :1]#(3567,12,170,1)

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])#x flow特征 均值方差归一化
    x_val[..., 0] = scaler.transform(x_val[..., 0])#
    x_test[..., 0] = scaler.transform(x_test[..., 0])#

    # y_train[..., 0] = scaler.transform(y_train[..., 0])
    # y_val[..., 0] = scaler.transform(y_val[..., 0])
    # y_test[..., 0] = scaler.transform(y_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)
    #转为tensor
    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )
    #dataloder
    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
