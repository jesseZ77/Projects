import numpy as np
import torch
import torch.nn as nn
import time
import pyro


def load_data_as_tensor(data_folder="data/", use_gpu=False):
    X_train = torch.tensor(np.load(data_folder + "X_train.npy")).type(torch.float)
    X_val = torch.tensor(np.load(data_folder + "X_val.npy")).type(torch.float)
    X_test = torch.tensor(np.load(data_folder + "X_test.npy")).type(torch.float)

    y_train = torch.tensor(np.load(data_folder + "y_train.npy")).type(torch.float)
    y_val = torch.tensor(np.load(data_folder + "y_val.npy")).type(torch.float)
    y_test = torch.tensor(np.load(data_folder + "y_test.npy")).type(torch.float)

    if use_gpu:
        return X_train.cuda(), X_val.cuda(), X_test.cuda(), y_train.cuda(), y_val.cuda(), y_test.cuda()
    return X_train, X_val, X_test, y_train, y_val, y_test


def gradient_descent(epochs, model, X_train, y_train, X_val, y_val, criterion=nn.MSELoss(),
                     patience=500, checkpoint_path='checkpoint.pt'):

    optimizer = torch.optim.Adam(model.parameters())
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    train_losses = []
    val_losses = []

    start = time.time()
    for i in range(epochs):

        model.train()
        y_pred_train = model(X_train)
        train_loss = criterion(y_pred_train, y_train)
        train_losses.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        y_pred_val = model(X_val)
        val_loss = criterion(y_pred_val, y_val)
        val_losses.append(val_loss.item())
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Stopping early")
            break

        if (i + 1) % 1000 == 0:
            print(f"Done: {i + 1} / {epochs} epochs")
            print(f"Train loss: {train_loss.item()}")
            print(f"Val loss: {val_loss.item()}")
            print(f"Time taken: {time.time() - start}s \n")

    return model, train_losses, val_losses, early_stopping


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    def __init__(self, patience=500, path='checkpoint.pt', bayes_model=False):
        self.patience = patience
        self.counter = 0
        self.best_score = -np.Inf
        self.early_stop = False
        self.path = path
        self.bayes_model = bayes_model

    def __call__(self, val_loss, model=None):
        score = -val_loss
        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.bayes_model:
            pyro.get_param_store().save(self.path)
        else:
            torch.save(model.state_dict(), self.path)



def variational_bayes():
    pass






