import numpy as np
import torch
import torch.nn as nn
import time
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro import poutine


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


def variational_bayes(epochs, bayes_model, guide, X_train, y_train, X_val, y_val,
                      patience=500, checkpoint_path='checkpoint.pt', num_particles=10):
    adam = pyro.optim.Adam({'lr': 0.001})
    svi = SVI(bayes_model, guide, adam, loss=Trace_ELBO(num_particles=num_particles))
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path, bayes_model=True)
    train_losses = []
    val_losses = []

    guide.requires_grad_(True)
    start = time.time()
    for i in range(epochs):
        train_loss = svi.step(X_train, y_train) / len(y_train)
        train_losses.append(train_loss)

        val_loss = svi.evaluate_loss(X_val, y_val) / len(y_val)
        val_losses.append(val_loss)

        early_stopping(val_loss, None)
        if early_stopping.early_stop:
            print("Stopping early")
            break

        if (i + 1) % 250 == 0:
            print(f"Done: {i + 1} / {epochs} epochs")
            print(f"Train loss: {train_loss}")
            print(f"Val loss: {val_loss}")
            print(f"Time taken: {time.time() - start}s \n")

    return train_losses, val_losses


def get_posterior_samples(predictive, X, y):
    with torch.no_grad():
        posterior_sample = predictive(X, y)
        posterior_losses = [nn.MSELoss()(y_pred, y).detach().item() for y_pred in posterior_sample['_RETURN']]
    return posterior_sample, posterior_losses


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    def __init__(self, patience=500, path='checkpoint.pt', bayes_model=False):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = np.Inf
        self.early_stop = False
        self.path = path
        self.bayes_model = bayes_model

    def __call__(self, val_loss, model=None):
        if val_loss > self.best_val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.bayes_model:
            pyro.get_param_store().save(self.path)
        else:
            torch.save(model.state_dict(), self.path)




