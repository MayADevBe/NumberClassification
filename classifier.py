import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import matplotlib.pyplot as plt

# TODO save laod model
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

#Path to save model
PATH = "model.pt"

class Classifier:

    def __init(self):
        self.net = None
        self.trainset = None
        self.testset = None


    def create(self):
        if os.path.isfile(PATH):
            self.load()
        else:
            print("Getting Data...")
            self.get_data()
            self.net = Net()
            print("Training Model...")
            self.train()
            print("Testing Model...")
            self.test()
            self.save()

    #save
    def save(self):
        torch.save(self.net.state_dict(), PATH)
        print("Model saved!")
    #load
    def load(self):
        self.net = Net()
        self.net.load_state_dict(torch.load(PATH))
        self.net.eval()
        print("Model loaded.")


    """DATA"""
    def get_data(self):
        train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))


        self.trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
        self.testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


    """Check Balance"""
    def check_balance(self):
        total = 0
        counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

        for data in self.trainset:
            Xs, ys = data
            for y in ys:
                counter_dict[int(y)] += 1
                total += 1

        print(counter_dict)

        for i in counter_dict:
            print(f"{i}: {counter_dict[i]/total*100.0}%")


    """TRAIN"""
    def train(self):
        # Optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        EPOCHS = 3

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}")
            for data in self.trainset:
                #data is a batch of featuressets and labels
                X, y = data
                self.net.zero_grad()
                output = self.net(X.view(-1, 28*28))
                # Loss
                loss = F.nll_loss(output, y) # cause output is vector
                loss.backward()
                optimizer.step()
            print(loss)


    """Test/Validation"""
    def test(self):
        correct = 0
        total = 0

        with torch.no_grad(): # out of sample data
            print("Validating...")
            for data in self.trainset:
                X, y = data
                output = self.net(X.view(-1, 28*28))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        print(f"Accuracy: {round(correct/total, 3)}")


    """Classification"""
    def classify(self, img):
        print("Classifying...")
        # turn image matrix
        print(img)
        img = list(zip(*img))
        print(img)
        x = torch.FloatTensor(img)
        with torch.no_grad(): # model shouldn't learn
            output = torch.argmax(self.net(x.view(-1, 28*28)))
        return output

    def show_img(self, tensor, output):
        plt.imshow(tensor.view(28,28))
        plt.title(output)
        plt.show()
        print("Showed")