#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset, Food101
from typing import List, Tuple
from flwr.common import Metrics

import flwr as fl
import numpy as np


# In[2]:


import os
import multiprocessing

data_path = os.path.join(os.getcwd(),'data', 'food-101')
cpu_count = multiprocessing.cpu_count() - 1 # set as you like!
#device = torch.device("mps") #CHANGE THIS TO FIT YOUR DEVICE PLEASE :D (maybe under fits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


pool_size = 4  # number of dataset partions (= number of total clients)

client_resources = {
        "num_cpus": cpu_count
}  # each client will get allocated 1 CPUs

transformations = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])


# Download Dataset
try:
    train_data = Food101(data_path, transform=transformations)
except:
    train_data = Food101(data_path, transform=transformations, download=True) 
test_data = Food101(data_path, split='test', transform=transformations)

lengths = []
while sum(lengths) != len(train_data):
    lengths = [round(x) for x in np.random.dirichlet(
        np.ones(pool_size),size=1)[0] * len(train_data)]
    
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32)
num_examples = {"trainset" : len(train_data), "testset" : len(test_data)}


# In[4]:


# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# In[5]:


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print("Loss: %f, Accuracy: %f" % (loss, accuracy))
    return loss, accuracy


# In[6]:


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 101)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and data
net = Net().to(device)


# In[7]:


class FlowerClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, 3, device)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader, device)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


# In[8]:


len(test_data.classes)


# ### Before you start training!
# 
# - Make sure your device is properly set above to fit your compute.
# - If you have made any changes to this script, download it as a python file and replace the flower/client.py file.
# - Open a separate terminal and run `python flower/server.py`.
# - Open 1-3 more terminals and run `python flower/client.py`.
# - Then run the following cell to also run a client here and watch! :)
# 
# If you want to change any of the model parameters, structure or even the splits on the data, you'll want to restart the server and clients. Have fun and experiment!

# In[ ]:


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)


# ## Challenges
# 
# - Adjust the fit and evaluate settings and see how the performance changes.
# - Try out another [Flower tutorial](https://flower.dev/docs/quickstart-pytorch.html).
# - Get a group of several folks together to try running flower in a distributed setup. Document your learnings and share in the reader-contributions!

# In[ ]:




