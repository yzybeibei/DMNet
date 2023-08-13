import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset import *
from model_complexcnn import *
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame

def test(model, test_dataloader):
    model.eval()
    correct = 0
    device = torch.device("cuda:0")
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

def Data_prepared(n_classes):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TestDataset_prepared(n_classes):
    X_test, Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    return X_test, Y_test

def main():
    X_test, Y_test = TestDataset_prepared(10)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load("model_weight/complexcnn_centerloss_n_classes_10.pth")
    test(model,test_dataloader)

if __name__ == '__main__':
   main()
