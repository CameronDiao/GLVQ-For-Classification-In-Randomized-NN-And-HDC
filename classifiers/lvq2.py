import numpy as np
import os
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel

import torch
from torch.utils.data import TensorDataset
from classifiers.pytorch import squared_euclidean_distance, kernel_distance
from classifiers.pytorch import GLVQLoss, KGLVQLoss
from classifiers.pytorch import Prototypes1D

from classifiers.pytorch import PyTorchObjective

def lvq2(inputs, labels, classifier, optimizer, epochs, ppc, beta, sigma=None):
    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels)

    if classifier == "glvq":
        model = glvq_module(inputs, labels, ppc=ppc)
        criterion = GLVQLoss(squashing="sigmoid_beta", beta=beta)
    elif classifier == "kglvq":
        model = kglvq_module(inputs, labels, ppc=ppc, sigma=sigma)
        criterion = KGLVQLoss(squashing='sigmoid_beta', beta=beta)
    else:
        raise ValueError("Invalid LVQ Classifier Type")

    if optimizer == "lbfgs":
        model = scipy_train(inputs, labels, model, criterion, ppc=ppc, iterations=2500)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=1e-4)
        batch_train(inputs, labels, model, optimizer, criterion, epochs)
        model.load_state_dict(torch.load(os.getcwd() + "/checkpoint.pt"))
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch_train(inputs, labels, model, optimizer, criterion, epochs)
        model.load_state_dict(torch.load(os.getcwd() + "/checkpoint.pt"))
    else:
        raise ValueError("Invalid Optimizer")

    return model


def glvq_module(x_data, y_data, ppc):
    class Model(torch.nn.Module):
        def __init__(self, x_data, y_data, **kwargs):
            super().__init__()
            self.p1 = Prototypes1D(input_dim=x_data.shape[1],
                                   prototypes_per_class=ppc,
                                   nclasses=torch.unique(y_data).size()[0],
                                   prototype_initializer='stratified_mean',
                                   data=[x_data, y_data])
            self.train_data = x_data

        def forward(self, x):
            protos = self.p1.prototypes
            plabels = self.p1.prototype_labels
            dis = squared_euclidean_distance(x, protos)
            return dis, plabels

    return Model(x_data=x_data, y_data=y_data)

def kglvq_module(x_data, y_data, ppc, sigma=None):
    class Model(torch.nn.Module):
        def __init__(self, x_data, y_data, **kwargs):
            super().__init__()
            self.p1 = Prototypes1D(input_dim=x_data.shape[1],
                                   prototypes_per_class=ppc,
                                   nclasses=torch.unique(y_data).size()[0],
                                   prototype_initializer='kernel_mean',
                                   data=[x_data, y_data])
            self.train_data = x_data

        def forward(self, x):
            protos = self.p1.prototypes
            plabels = self.p1.prototype_labels
            dis = kernel_distance(torch.from_numpy(rbf_kernel(x, gamma=sigma)),
                                  torch.from_numpy(rbf_kernel(x, self.train_data, gamma=sigma)),
                                  torch.from_numpy(rbf_kernel(self.train_data, gamma=sigma)), x, protos)
            return dis, plabels

    return Model(x_data=x_data, y_data=y_data)

def full_train(x_data, y_data, model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        def closure():
            optimizer.zero_grad()
            distances, plabels = model(x_data)
            loss = criterion([distances, plabels], y_data)
            print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')
            loss.backward()
            return loss

        optimizer.step(closure)
    torch.save(model.state_dict(), os.getcwd() + '/checkpoint.pt')

def batch_train(x_data, y_data, model, optimizer, criterion, epochs, scheduler=None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    trainloader = torch.utils.data.DataLoader(TensorDataset(x_data, y_data), batch_size=16, num_workers=0,
                                              shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            distances, plabels = model(inputs)
            loss = criterion([distances, plabels.to(device)], targets)
            #print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
    torch.save(model.state_dict(), os.getcwd() + '/checkpoint.pt')

def scipy_train(x_data, y_data, model, loss, ppc, iterations=2500):
    nb_classes = torch.unique(y_data).size()[0]
    nb_features = x_data.shape[1]
    obj = PyTorchObjective(model, loss, x_data, y_data)
    res = minimize(fun=obj.fun, jac=obj.jac, method='l-bfgs-b', x0=obj.x0,
                   options={'gtol': 1e-5, 'maxiter': iterations})
    return res.x.reshape((ppc * nb_classes, nb_features))

