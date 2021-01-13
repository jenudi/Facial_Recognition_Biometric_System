import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
from torch.nn import functional as F


class DataArgs:
    def __init__(self, csv_path='mini data/datasets/'):
        self.csv_path = csv_path
        self.train = self.csv_path + 'train.csv'
        self.val = self.csv_path + 'validation.csv'
        self.test = self.csv_path + 'test.csv'
        self.embedding = 'embedding'
        self.target = 'name'

        # Hyper-parameters
        self.in_channel = 1
        self.learning_rate = 1e-3
        self.train_batch_size = 100
        self.num_epochs = 50


# Set Dataset
class TabularDataset(Dataset):
    def __init__(self, csv_file, transform=None):  # train=False
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, index):
        value = self.csv_file.iloc[index, 1]
        if self.transform:
            value = self.transform(value)
        y = torch.tensor(self.csv_file.iloc[index, 0])
        y = y.to(torch.float32)
        return value, y

    def __len__(self):
        return len(self.csv_file)


# Model
class Net(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        pass

    def forward(self, x):
        pass


args = DataArgs()
train_dataset = TabularDataset(csv_file=args.csv_path, transform= transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)


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
