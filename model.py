import torch
import torch.nn.functional as F
import torch.nn as nn
import helper_fns
import time
from torchnet.meter import ClassErrorMeter, AverageValueMeter
import prototypical_network
from torch_prototypes.metrics import distortion, cost
from torch_prototypes.metrics.distortion import DistortionLoss
from  torch.distributions import multivariate_normal
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Pytorch version of the 3 fully connected layers
# No problem
class Net(nn.Module):
    def __init__(self, mode, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size):
        super(Net, self).__init__()
        self.mode = mode
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# No problem
class PL(nn.Module):
    def __init__(self, centers, weights, vars):
        super(PL, self).__init__()
        self.centers = centers
        self.weights = weights
        self.vars = vars

    def forward(self, mapping, labels):
        # Find prototype by labels
        targets = torch.index_select(self.centers, 0, labels)
        # Sum the distance between each point and its prototype
        weights = torch.index_select(self.weights, 0, labels)
        log_vars = torch.log(torch.index_select(self.vars, 0, labels))
        # dist = torch.norm(mapping - targets, dim=1)/weights
        # return torch.sum(dist)/mapping.shape[0]
        likelihood = -helper_fns.log_likelihood_student(mapping, targets, log_vars)/weights
        return torch.sum(likelihood)/mapping.shape[0]
        # return torch.sum(likelihood)/mapping.shape[0]


def train(mode, loss_mode, epochs, embedding_dim, D, num_celltypes, encoder, dataset, dataloader_training, dataloader_testing, obs_name, init_weights):
    if torch.cuda.is_available():
        D_metric = D.cuda()
    else:
        D_metric = D
    # A simple neural network no problem
    if mode == 'Net':
        if torch.cuda.is_available():
            model = Net(mode, 128, 32, 37).cuda()
        else:
            model = Net(mode, 128, 32, 37)
    # Learnt prototype & Simple neural network encoder no problem
    elif mode == 'Proto_Net':
        if torch.cuda.is_available():
            model = Net(mode, 128, 32, embedding_dim).cuda()
            centers = []
            vars = []
            for i in range(len(num_celltypes)):
                out = model(torch.tensor(dataset[dataset.obs[obs_name] == encoder.inverse_transform([i])[0]].X))
                centers.append(np.array(torch.mean(out, dim=0)))
                vars.append(np.array(torch.var(out, dim=0)))
            centers = torch.tensor(centers, dtype=float).cuda()
            vars = torch.tensor(vars, dtype=float).cuda()
            model = prototypical_network.LearntPrototypes(model, n_prototypes= D.shape[0],
                    prototypes=centers, vars=vars, embedding_dim=embedding_dim, device='cuda').cuda()
        else:
            model = Net(mode, 128, 64, 32, 16, embedding_dim)
            centers = []
            vars = []
            for i in range(len(num_celltypes)):
                out = model(torch.tensor(dataset[dataset.obs[obs_name] == encoder.inverse_transform([i])[0]].X))
                centers.append(np.array(torch.mean(out, dim=0).detach()))
                vars.append(np.array(torch.var(out, dim=0).detach()))
            centers = torch.tensor(centers, dtype=float)
            vars = torch.tensor(vars, dtype=float)
            model = prototypical_network.LearntPrototypes(model, n_prototypes= D.shape[0],
                    prototypes=centers, vars=vars, embedding_dim=embedding_dim, device='cpu')
    # Cross entropy loss no problem
    criterion = nn.CrossEntropyLoss()
    # Distortion loss no problem
    delta = DistortionLoss(D_metric)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Train & Test model, no problem
    for epoch in range(1, epochs+1):
        print('Epoch {}'.format(epoch))
        ER_meter = ClassErrorMeter(accuracy=False)

        model.train()
        t0 = time.time()
        for batch in dataloader_training:
            if torch.cuda.is_available():
                x = batch.X.cuda()
                y = batch.obs[obs_name].type(torch.LongTensor).cuda()
            else:
                x = batch.X
                y = batch.obs[obs_name].type(torch.LongTensor)
            y = y.squeeze()
            y.long()
            if mode == 'Net':
                out = model(x)
            elif mode == 'Proto_Net':
                out, embeddings = model(x)
            opt.zero_grad()
            xe_loss = criterion(out, y)
            loss = xe_loss
            if 'pl' in loss_mode:
                pl_loss = PL(centers = model.prototypes.data, weights=init_weights, vars=model.vars)
                pl_loss_ = pl_loss(embeddings, y)
                loss = loss + 1000*pl_loss_
            if 'disto' in loss_mode:
                disto_loss = delta(model.prototypes)
                loss = loss + disto_loss
            loss.backward()
            opt.step()
            pred = out.detach()
            ER_meter.add(pred.cpu(),y.cpu())
        vars = []
        if mode == 'Proto_Net':
            if torch.cuda.is_available():
                for i in range(len(num_celltypes)):
                    out, embeddings = model(torch.tensor(dataset[dataset.obs[obs_name] == encoder.inverse_transform([i])[0]].X)).cpu()
                    vars.append(np.array(torch.var(embeddings, dim=0).detach().cpu()))
                model.vars = torch.tensor(vars, dtype=float).cuda()
            else:
                for i in range(len(num_celltypes)):
                    out, embeddings = model(torch.tensor(dataset[dataset.obs[obs_name] == encoder.inverse_transform([i])[0]].X))
                    vars.append(np.array(torch.var(embeddings, dim=0).detach()))
                model.vars = torch.tensor(vars, dtype=float)
        t1 = time.time()
        
        print('Train ER {:.2f}, time {:.1f}s'.format(ER_meter.value()[0], t1-t0))

        model.eval()
        ER_meter = ClassErrorMeter(accuracy=False)
        t0 = time.time()
        for batch in dataloader_testing:
            if torch.cuda.is_available():
                x = batch.X.cuda()
                y = batch.obs[obs_name].type(torch.LongTensor).cuda()
            else:
                x = batch.X
                y = batch.obs[obs_name].type(torch.LongTensor)
            y = y.squeeze()
            y.long()
            if mode == 'Net':
                with torch.no_grad():
                    out = model(x)
            elif mode == 'Proto_Net':
                with torch.no_grad():
                    out, embeddings = model(x)
            pred = out.detach()
            ER_meter.add(pred.cpu(),y)
        t1 = time.time()
        print('Test ER {:.2f}, time {:.1f}s'.format(ER_meter.value()[0], t1-t0))
    return model


# logistic regression
def train_logistic_regression(dataset, train_indices, test_indices, obs_name, encoder):
    X_train = dataset.X[train_indices]
    y_train = encoder.transform(dataset.obs[obs_name][train_indices])
    X_test = dataset.X[test_indices]
    y_test = encoder.transform(dataset.obs[obs_name][test_indices])
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    print('Logistic Regression')
    print('Train error: {}%'.format((1 - clf.score(X_train, y_train))*100))
    print('Test error: {}%'.format((1 - clf.score(X_test, y_test))*100))
    return clf


# kNN
def train_knn(dataset, train_indices, test_indices, obs_name, encoder):
    X_train = dataset.X[train_indices]
    y_train = encoder.transform(dataset.obs[obs_name][train_indices])
    X_test = dataset.X[test_indices]
    y_test = encoder.transform(dataset.obs[obs_name][test_indices])
    clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    print('kNN')
    print('Train error: {}%'.format((1 - clf.score(X_train, y_train))*100))
    print('Test error: {}%'.format((1 - clf.score(X_test, y_test))*100))
    return clf