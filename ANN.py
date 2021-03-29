import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
from torch.nn import functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from images_classes import *
#from aug import *
from tqdm import tqdm
import torch
#from new_main import train_df,validation_df
import albumentations as A
from albumentations.pytorch import ToTensor
from PIL import Image, ImageFile
#from new_main import sampler


class NewNet(nn.Module):
    def __init__(self, num_classes=1):
        super(NewNet,self).__init__()
        self.num_classes = num_classes
        self.model = InceptionResnetV1(classify=True,pretrained='vggface2', num_classes=self.num_classes)
        self.drop = nn.Dropout(0.5)
        self.head_linear = nn.Linear(self.num_classes,self.num_classes,bias=True)
        self.change_model()

    def forward(self, x):

        x = self.model(x)
        x = self.drop(x)
        x = self.head_linear(x)
        return x

    def change_model(self):
        self.model.dropout.p = 0.6
        self.model.last_linear.out_features = self.num_classes
        self.model.last_bn.num_features = self.num_classes
        self.model.logits.in_features = self.num_classes


class FRBSDataset(Dataset):
    def __init__(self, values, train_set=True):  # train=False
        self.values = values
        self.train_set = train_set
        self.norm = A.Compose([A.Resize(160, 160, interpolation=1, always_apply=True, p=1),
                               A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ), ToTensor(),])

    def __getitem__(self, index):  # 0: 'path', 1: 'cls', 2: 'face_indexes', 3 'aug'

        indexes = self.values[index][2]
        img = np.array(Image.open(self.values[index][0]))
        img = img[indexes[1]:indexes[3], indexes[0]:indexes[2]]
        if self.train_set and self.values[index][3]:
            img = aug_img(img)
        augmentations = self.norm(image=img)
        img = augmentations["image"]
        cls = torch.tensor(self.values[index][1])
        return img, cls

    def __len__(self):
        return len(self.values)


class Ann:
    def __init__(self, batch_size=100, epochs=10, lr=1e-3, l2=0.01):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print(self.device)
        self.train_dl = None
        self.val_dl = None
        self.num_workers = 0
        self.init_dls()
        self.model = NewNet(num_classes=self.out_features).to(self.device)
        self.optimizer = self.optimizer()
        self.training_loss = 0.0
        self.val_loss = 0.0
        self.all_training_loss = list()
        self.all_val_loss = list()
        self.y_pred = list()
        self.y_pred_proba = list()

    def save_model(self, filename="drive/MyDrive/amdocs_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="drive/MyDrive/amdocs_model.pth"):
        self.model.load_state_dict(torch.load(filename))

    def save_checkpoint(self, filename="drive/MyDrive/amdocs_model.pth"):
        checkpoint = {"state_dict": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict()}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint_file="drive/MyDrive/amdocs_model.pth", ):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

    def init_dls(self):
        temp_pd = train_df
        self.out_features = len(temp_pd['class'].value_counts())
        train_dataset = FRBSDataset(values=temp_pd.values, )
        self.train_dl = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                   pin_memory=self.use_cuda, sampler=sampler)
        temp_pd = validation_df
        val_dataset = FRBSDataset(values=temp_pd.values, train_set=False)
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                 pin_memory=self.use_cuda)

    def main(self, decay_learning=False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.05)
        print('Start training')
        for epoch_ndx in range(self.epochs):
            print(f"\n epoch: {epoch_ndx}")
            if decay_learning:
                self.lr_schedule(epoch_ndx)
            trn_loss = self.training(epoch_ndx, self.train_dl)
            self.all_training_loss.append(trn_loss.detach() / len(self.train_dl))
            val_loss = self.validation(epoch_ndx, self.val_dl)
            self.all_val_loss.append(val_loss.detach() / len(self.val_dl))
        print("Finished")
        return self.y_pred, self.y_pred_proba

    def training(self, epoch_ndx, train_dl):
        self.model.train()
        self.training_loss = 0.0
        loop = tqdm(train_dl, position=0, leave=True)
        for batch_ndx, batch_tup in enumerate(loop, 0):
            self.optimizer.zero_grad()
            loss, _, _ = self.compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size)
            self.training_loss += loss
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.item())
        return self.training_loss

    def validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            self.val_loss = 0.0
            loop = tqdm(val_dl, position=0, leave=True)
            for batch_ndx, batch_tup in enumerate(loop, 0):
                loss, proba, pred = self.compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size)
                loop.set_postfix(loss=loss.item())
                self.val_loss += loss
                if (epoch_ndx + 1) % self.epochs == 0:
                    for i in pred:
                        self.y_pred.append(int(i))
                    for i in proba:
                        self.y_pred_proba.append(round(float(i), 2))
        return self.val_loss

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size):
        input, cls = batch_tup
        input = input.to(self.device, dtype=torch.float)
        cls = torch.reshape(cls, (-1,))
        cls = cls.type(torch.LongTensor)
        cls = cls.to(self.device)
        logits_g = self.model(input)
        loss_func = nn.CrossEntropyLoss(reduction='none')  # weight=self.class_weights
        loss_g = loss_func(logits_g, cls)
        return loss_g.mean(), torch.max(F.softmax(logits_g.detach(), dim=1), 1)[0], \
               torch.max(F.softmax(logits_g.detach(), dim=1), 1)[1]


#a = Ann(batch_size=300, epochs=20, lr=0.0001, l2=0.001)
#%%
#y_pred, y_proba = a.main()

#y_true = pd.read_csv(args.val)
#print(f1_score(y_true['id'], l, average='micro'))
