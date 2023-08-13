import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn import *
from get_dataset import TrainDataset, TestDataset
from center_loss import CenterLoss

def train(model, loss_nll, loss_center, train_dataloader, optimizer_model, optimizer_cent, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    result_loss = 0
    nll_loss = 0
    cent_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer_model.zero_grad()
        optimizer_cent.zero_grad()
        output = model(data)
        classifier_value = F.log_softmax(output[1], dim=1)
        nll_loss_batch = loss_nll(classifier_value, target)
        cent_loss_batch = loss_center(output[0], target)
        weight_cent = 0.01
        result_loss_batch = nll_loss_batch + weight_cent * cent_loss_batch
        result_loss_batch.backward()
        optimizer_model.step()
        for param in loss_center.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_cent.step()

        nll_loss += nll_loss_batch.item()
        cent_loss += cent_loss_batch.item()
        result_loss += result_loss_batch.item()
        pred = classifier_value.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    nll_loss /= len(train_dataloader)
    cent_loss /= len(train_dataloader)
    result_loss /= len(train_dataloader)

    print('Train Epoch: {} \tClassifier_Loss: {:.6f}, Center_Loss, {: 6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        nll_loss,
        cent_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', result_loss, epoch)

def test(model, loss_model, test_dataloader, epoch, writer, device_num):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            output = F.log_softmax(output[1], dim=1)
            test_loss += loss_model(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    writer.add_scalar('Accuracy', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Loss/test', test_loss,epoch)

    return test_loss

def train_and_test(model, loss_model, loss_center, train_dataloader, val_dataloader, optimizer_model, optimizer_cent, epochs, writer, save_path, device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_model, loss_center, train_dataloader, optimizer_model, optimizer_cent, epoch, writer, device_num)
        test_loss = test(model, loss_model, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

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

def TrainDataset_prepared(n_classes):
    X_train, X_val,  value_Y_train, value_Y_val = TrainDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes)

    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])

    return  X_train, X_val, value_Y_train, value_Y_val

class Config:
    def __init__(
        self,
        batch_size: int = 16,
        test_batch_size: int = 16,
        epochs: int = 150,
        lr_model: float = 0.01,
        lr_cent: float = 0.5,
        log_interval: int = 10,
        n_classes: int = 10,
        save_path: str = 'model_weight/complexcnn_centerloss_n_classes_10.pth',
        device_num: int = 0,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_model = lr_model
        self.lr_cent = lr_cent
        self.log_interval = log_interval
        self.n_classes = n_classes
        self.save_path = save_path
        self.device_num = device_num

def main():
    conf = Config()
    writer = SummaryWriter("logs")
    device = torch.device("cuda:"+str(conf.device_num))

    X_train, X_val, value_Y_train, value_Y_val = TrainDataset_prepared(conf.n_classes)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

    model = base_complex_model()
    if torch.cuda.is_available():
        model = model.to(device)
    print(model)

    loss_nnl = nn.NLLLoss()
    if torch.cuda.is_available():
        loss_nnl = loss_nnl.to(device)

    use_gpu = torch.cuda.is_available()
    loss_center = CenterLoss(num_classes = conf.n_classes, feat_dim = 1024, use_gpu = use_gpu)

    optim_model = torch.optim.Adam(model.parameters(), lr=conf.lr_model)
    optim_centloss = torch.optim.Adam(loss_center.parameters(),lr= conf.lr_cent)

    train_and_test(model, loss_model=loss_nnl, loss_center = loss_center, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer_model=optim_model, optimizer_cent=optim_centloss, epochs=conf.epochs, writer=writer, save_path=conf.save_path, device_num=conf.device_num)

if __name__ == '__main__':
   main()