import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
#import matplotlib.pyplot as plt

"""Create Model"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # NN Layers
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Activation Functions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        #exit()
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

#Path to save model
PATH = "model.pt"
#Train with GPU if available
gpu = torch.cuda.is_available()

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
            if gpu:
                self.net = self.net.cuda()
            print("Training Model...")
            self.train()
            print("Testing Model...")
            self.test()
            self.save()

    #save
    def save(self, addition=""):
        torch.save(self.net.state_dict(), addition+PATH)
        print("Model saved!")
    #load
    def load(self):
        self.net = Net()
        self.net.load_state_dict(torch.load(PATH))
        self.net.eval()
        print("Model loaded.")


    """DATA"""
    def get_data(self):
        train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]))
        test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]))

        kwargs = {'num_workers': 0}#, 'pin_memory': True
        self.trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, **kwargs)
        self.testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True, **kwargs)


    """TRAIN"""
    def train(self):
        # Optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        criterion = F.nll_loss

        EPOCHS = 3

        for epoch in range(1, EPOCHS+1):
            running_loss = 0.0
            for batch_id, data in enumerate(self.trainset):
                #data is a batch of featuressets and labels
                X, y = data
                if gpu:
                    X = X.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                output = self.net(X)
                # Loss
                loss = criterion(output, y) # cause output is vector
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if batch_id % 400 == 399: 
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_id * len(data), len(self.trainset.dataset), 
                        100. * batch_id / len(self.trainset), running_loss / 400))
                    running_loss = 0.0
            #self.save(f"{epoch}") - save epochs individual


    """Test/Validation"""
    def test(self):
        correct = 0
        total = 0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        with torch.no_grad(): # out of sample data
            for data in self.trainset:
                X, y = data
                if gpu:
                    X = X.cuda()
                    y = y.cuda()
                output = self.net(X)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                c = (predicted == y).squeeze()
                for i in range(4):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))


    """Classification"""
    def classify(self, img):
        print("Classifying...")
        # turn image matrix
        img = [[img[j][i] for j in range(len(img))] for i in range(len(img[0]))]
        x = torch.FloatTensor(img)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        with torch.no_grad(): # model shouldn't learn
            output = torch.argmax(self.net(x))
        #self.show_img(x, output)
        return output

    # def show_img(self, tensor, output):
    #     plt.imshow(tensor.view(28,28))
    #     plt.title(output)
    #     plt.show()
    #     print("Showed") 

#train
classifier = Classifier()
classifier.create()