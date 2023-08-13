import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(num):
    x = np.load(f"../Dataset/X_train_{num}Class.npy")
    y = np.load(f"../Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.1, random_state=30)
    return X_train, X_val, Y_train, Y_val

def TrainDatasetKshotRround_DA(num, k, num_slice, num_DA):
    '''

    :param num: 类别数
    :param k:
    :param num_slice:
    :param num_DA:
    :return:
    '''
    x = np.load(f"Dataset/X_train_{num}Class.npy")
    y = np.load(f"Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)

    List_train = y.tolist()

    X_train_K_Shot_DA = np.zeros([int(k * num * num_DA), 6000, 2])
    Y_train_K_Shot_DA = np.zeros([int(k * num * num_DA)])

    for i in range(num):
        index_train_start = List_train.index(i)
        if i == num - 1:
            index_train_end = y.shape[0]
        else:
            index_train_end = List_train.index(i + 1) - 1
        index_shot = range(index_train_start, index_train_end)
        random_shot = random.sample(index_shot, k)

        x_ = x[random_shot, :, :]
        x_ = x_.reshape(k, num_slice, int(6000 / num_slice), 2)
        x_ = np.transpose(x_, (1, 0, 2, 3))

        x_DA = np.zeros([int(k * num_DA), 6000, 2])

        for j in range(num_DA):
            np.random.shuffle(x_)
            temp = x_
            temp = np.transpose(temp, (0, 1, 2, 3))
            temp = temp.reshape(k, 6000, 2)
            x_DA[j * k:j * k + k, :, :] = temp

        X_train_K_Shot_DA[i * k * num_DA:i * k * num_DA + k * num_DA, :, :] = x_DA
        y_ = y[random_shot]
        y_DA = np.tile(y_, num_DA)
        Y_train_K_Shot_DA[i * k * num_DA:i * k * num_DA + k * num_DA] = y_DA.reshape(-1)
    return X_train_K_Shot_DA, Y_train_K_Shot_DA

def TrainDatasetKShotRound(num,k):
    x = np.load(f"../Dataset/X_train_{num}Class.npy")
    y = np.load(f"../Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    List_train = y.tolist()

    X_train_K_Shot = np.zeros([int(k * num), 6000, 2])
    Y_train_K_Shot = np.zeros([int(k * num)])

    for i in range(num):
        index_train_start = List_train.index(i)
        if i == num - 1:
            index_train_end = y.shape[0]
        else:
            index_train_end = List_train.index(i + 1) - 1
        index_shot = range(index_train_start, index_train_end)
        random_shot = random.sample(index_shot, k)

        X_train_K_Shot[i * k:i * k + k, :, :] = x[random_shot, :, :]
        Y_train_K_Shot[i * k:i * k + k] = y[random_shot]
    return X_train_K_Shot, Y_train_K_Shot

def TestDataset(num):
    x = np.load(f"../Dataset/X_test_{num}Class.npy")
    y = np.load(f"../Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y
