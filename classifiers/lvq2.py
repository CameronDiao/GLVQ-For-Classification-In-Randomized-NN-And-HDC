import numpy as np

import torch
from torch.utils.data import TensorDataset
from sklearn.metrics.pairwise import rbf_kernel
from classifiers.pytorch import euclidean_distance, kernel_distance
from classifiers.pytorch import GLVQLoss, KGLVQLoss
from classifiers.pytorch import Prototypes1D

def lvq2(inputs, labels, classifier, optimizer, ppc, beta, sigma=None):
    inputs = torch.from_numpy(inputs).float()
    labels = torch.from_numpy(labels).float()

    if classifier == "glvq":
        model = glvq_module(inputs, labels, ppc=ppc)
        criterion = GLVQLoss(squashing="sigmoid_beta", beta=beta)
    elif classifier == "kglvq":
        model = kglvq_module(inputs, labels, ppc=ppc, sigma=sigma)
        criterion = KGLVQLoss(squashing='sigmoid_beta', beta=beta)
    else:
        raise ValueError("Invalid LVQ Classifier Type")

    if optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters())
        full_train(inputs, labels, model, optimizer, criterion)
        model.load_state_dict(torch.load("/Users/camerondiao/Documents/HDResearch/DataManip/checkpoint.pt"))
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
        batch_train(inputs, labels, model, optimizer, criterion)
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
            dis = euclidean_distance(x, protos)
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

def full_train(x_data, y_data, model, optimizer, criterion):
    best_loss = np.inf

    for epoch in range(50):
        model.train()
        def closure():
            optimizer.zero_grad()
            distances, plabels = model(x_data)
            loss = criterion([distances, plabels], y_data)
            # print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')
            loss.backward()
            return loss

        optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            distances, plabels = model(x_data)
            loss = criterion([distances, plabels], y_data)

        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), 'checkpoint.pt')

def batch_train(x_data, y_data, model, optimizer, criterion):
    trainloader = torch.utils.data.DataLoader(TensorDataset(x_data, y_data), batch_size=128, num_workers=0,
                                              shuffle=True)

    for epoch in range(50):
        model.train()
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            distances, plabels = model(inputs)
            loss = criterion([distances, plabels], targets)
            print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')
            loss.backward()
            optimizer.step()
