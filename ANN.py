import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
from torch.nn import functional as F
import ast
import numpy as np

class DataArgs:
    def __init__(self, csv_path='sets_csv_files/'):
        self.csv_path = csv_path
        self.train = self.csv_path + 'train.csv'
        self.val = self.csv_path + 'validation.csv'
        self.test = self.csv_path + 'test.csv'
        self.embedding = 'embedding'
        self.target = 'name'

        self.num_features= 1
        self.num_in_features = 100

        # Hyper-parameters
        self.in_channel = 1
        self.learning_rate = 1e-3
        self.train_batch_size = 5
        self.num_epochs = 50


# Set Dataset
class TabularDataset(Dataset):
    def __init__(self, values, transform=None):  # train=False
        self.values = values
        self.transform = transform

    def __getitem__(self, index):
        x = self.values[index][2]
        x = eval(x)
        x = torch.tensor(x)
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(self.values[index, 0])
        y = y.to(torch.float32)
        return x, y

    def __len__(self):
        return len(self.values)


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


args = DataArgs()
train_dataset = TabularDataset(values=pd.read_csv(args.train).values, transform=None)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)


class Ann:
    def __init__(self, args, train_loader):
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = Net()
        self.optimizer = self.optimizer()
        self.train_loader = train_loader

    def optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.99)

    def main(self):
        print('Starting')
        for epoch_ndx in range(1, self.args.num_epochs + 1):
            self.training(epoch_ndx)
        print("Finished: Ranzcr.main()")


    def training(self, epoch_ndx):
        self.model.train()
        for batch_ndx, batch_tup in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss = self.compute_batch_loss(batch_tup)
            print(f"E: {epoch_ndx}, batch: {batch_ndx}, loss: {loss}")
            loss.backward()
            self.optimizer.step()

    def compute_batch_loss(self, batch_tup):
        # self.train_loader.batch_size
        input_t, label_t = batch_tup
        input_g = input_t.to(self.device)
        label_g = label_t.to(self.device)
        logits_g = self.model(input_g) #probability_g
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g)
        return loss_g.mean()





#%% Train Function
def training_loop(n_epochs, optimizer, model, loss_fn, loader):
    for epoch in range(n_epochs):
        model.train()
        loss_train = 0.0
        for index, (inputs, labels) in enumerate(loader):
            # forward
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            print(f'epoch: {epoch + 1} , batch: {index} , Train loss: {loss_train/len(loader)} , loss: {loss}')


#%% Run the model
model = Net()
training_loop(n_epochs=args.num_epochs,
              model=model,
              optimizer=optim.SGD(model.parameters(),
                                  lr=args.learning_rate),
              loss_fn=nn.MSELoss(),
              loader=train_loader)


def validate(model, loaders):
    for name, loader in [("train", loaders[0]), ("val", loaders[1])]:
        num_correct, num_samples = 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)

                labels = labels.cpu().numpy()
                _, predicted = torch.max(outputs, dim=1)
                num_correct += sum([1 for index, value in enumerate(predicted) if list(value) == list(labels[index])])
                num_samples += labels.shape[0]

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")


val_dataset = TobularDataset(csv_file=args, transform= transforms.ToTensor())
val_loader = DataLoader(dataset=val_dataset, shuffle=False)

validate(model,[train_loader,val_loader])