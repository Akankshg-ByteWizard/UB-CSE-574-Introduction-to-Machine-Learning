'''
Comparing single layer MLP with deep MLP (using PyTorch)
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron():

    class net(nn.Module):
        def __init__(self):
            super().__init__()

            # Network Parameters
            n_hidden_1 = 256  # 1st layer number of features
            n_hidden_2 = 256  # 2nd layer number of features
            n_input = 2376  # data input
            n_classes = 2

            # Initialize network layers
            self.layer_1 = nn.Linear(n_input, n_hidden_1)
            self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
            self.out_layer = nn.Linear(n_hidden_2, n_classes)

        def forward(self, x):
            x = F.relu(self.layer_1(x))
            x = F.relu(self.layer_2(x))
            x = self.out_layer(x)
            return x

    return net()

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = np.squeeze(labels)
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]

    class dataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    trainset = dataset(train_x, train_y)
    validset = dataset(valid_x, valid_y)
    testset = dataset(test_x, test_y)

    return trainset, validset, testset


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Parameters
learning_rate = 0.0001
training_epochs = 50
batch_size = 100

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Construct model
model = create_multilayer_perceptron().to(device)

# Define loss and openptimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# load data
trainset, validset, testset = preprocess()
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# Training cycle
for t in range(training_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, cost, optimizer)
print("Optimization Finished!")
test(test_dataloader, model, cost)