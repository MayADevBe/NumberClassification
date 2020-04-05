import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# TODO make class with attributes + classify method

"""DATA"""
def get_data():
    global trainset, testset
    train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))


    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


"""Visualisation"""
def visualize():
    for data in trainset:
        plt.imshow(data[0][0].view(28,28))
        print(data[1][0])
        plt.show()
        break

"""Check Balance"""
def check_balance():
    total = 0
    counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    for data in trainset:
        Xs, ys = data
        for y in ys:
            counter_dict[int(y)] += 1
            total += 1

    print(counter_dict)

    for i in counter_dict:
        print(f"{i}: {counter_dict[i]/total*100.0}%")


"""Create Model"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # NN Layers
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # Activation Functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def create():
    global net
    net = Net()
    #print(net)


"""TRAIN"""
def train():
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    EPOCHS = 3

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        for data in trainset:
            #data is a batch of featuressets and labels
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 28*28))
            # Loss
            loss = F.nll_loss(output, y) # cause output is vector
            loss.backward()
            optimizer.step()
        print(loss)


"""Test/Validation"""
def test():
    correct = 0
    total = 0

    with torch.no_grad(): # out of sample data
        print("Validating...")
        for data in trainset:
            X, y = data
            output = net(X.view(-1, 28*28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print(f"Accuracy: {round(correct/total, 3)}")


# """Input own Data"""

# o = torch.argmax(net(X[0].view(-1, 28*28))[0])

# plt.imshow(X[0].view(28,28), cmap='Greys')
# plt.title(f"{o}")
# plt.show()

# """Pass Data Example"""
# X = torch.rand((28,28))
# X = X.view(-1, 28*28)
# output = net(X)
# print(output)



