import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
from torch.nn import functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from images_classes import *
from aug import *
from tqdm import tqdm
import torch
import datetime
from new_main import train_df,validation_df


class FRBSDataset(Dataset):
    def __init__(self, values,train_set=True):  # train=False
        self.values = values
        self.train_set = train_set

    def __getitem__(self, index): # 0: 'path', 1: 'cls', 2: 'face_indexes', 3 'aug'

        image_class = torch.tensor(self.values[index][1])

        image = Image.open(self.values[index][0])
        face_image = image.crop(self.values[index][2])
        if self.train_set and self.values[index][3]:
            aug_image = aug_img(face_image)
        else:
            aug_image = aug_img2(face_image)
        norm_image = cv.normalize(aug_image, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        resized_img = torch.tensor(norm_image)
        resized_img = resized_img.permute(2, 0, 1)
        return resized_img, image_class

    def __len__(self):
        return len(self.values)

class Ann:
    def __init__(self, batch_size = 100,epochs = 10,lr = 1e-3,l2=0.01,save_model=False,load_model=False):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_model = save_model
        self.load_model = load_model
        self.l2 = l2
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print(self.device)
        self.train_dl = None
        self.val_dl = None
        self.num_workers = 0
        self.init_dls()
        self.model = InceptionResnetV1(classify=True,pretrained='vggface2', num_classes=self.out_features).to(self.device)
        #self.model = Densenet().cuda()
        self.optimizer = self.optimizer()
        self.training_loss = 0.0
        self.val_loss = 0.0
        self.all_training_loss = list()
        self.all_val_loss = list()
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.y_pred = list()
        self.y_pred_proba = list()

    def optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=self.l2)
        #return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)

    def lr_schedule(self,epoch):
      if (epoch + 1) % 8 == 0:
        self.lr *= 0.96
        print(self.lr)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99,weight_decay=self.l2)

    def init_dls(self):
        temp_pd = train_df
        self.out_features = len(temp_pd['class'].value_counts())
        train_dataset = FRBSDataset(values=temp_pd.values,)
        self.train_dl = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=self.use_cuda)
        #sorted_dict = collections.OrderedDict(sorted(temp_pd['class'].value_counts().to_dict().items()))
        temp_pd = validation_df
        val_dataset = FRBSDataset(values=temp_pd.values,train_set=False)
        self.val_dl = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False,
                                     num_workers=self.num_workers,pin_memory=self.use_cuda)

    def main(self,decay_learning=False):
        if self.load_model is not False:
            try:
                self.model.load_state_dict(torch.load(self.load_model))
                print('Loaded model for training')
            except FileNotFoundError:
                print('File Not Found')
                return
        else:
            print('Start training')
        for epoch_ndx in range(self.epochs):
            print(f"epoch: {epoch_ndx}")
            if decay_learning:
                self.lr_schedule(epoch_ndx)
            trn_loss = self.training(epoch_ndx, self.train_dl)
            self.all_training_loss.append(trn_loss.detach() / len(self.train_dl))
            val_loss = self.validation(epoch_ndx, self.val_dl)
            self.all_val_loss.append(val_loss.detach() / len(self.val_dl))
            if self.save_model:
              torch.save(self.model.state_dict(), f'drive/MyDrive/{self.time}_model.pth')
        plt.figure()
        plt.plot(self.all_training_loss, 'r', label="Train")
        plt.plot(self.all_val_loss, 'b', label="Validation")
        plt.suptitle("Model: {self.time} lr: {self.lr}")
        plt.legend(loc="upper right")
        plt.xlabel('epochs')  # fontsize=18
        plt.ylabel('loss')
        plt.show()
        print("Finished")
        return self.y_pred, self.y_pred_proba

    def training(self, epoch_ndx, train_dl):
        self.model.train()
        self.training_loss = 0.0
        loop = tqdm(train_dl, position=0,leave=True)
        for batch_ndx, batch_tup in enumerate(loop,0):
            self.optimizer.zero_grad()
            loss, _,_ = self.compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size)
            self.training_loss += loss
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.item())
        return self.training_loss

    def validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            self.val_loss = 0.0
            loop = tqdm(val_dl, position=0,leave=True)
            for batch_ndx, batch_tup in enumerate(loop, 0):
                loss, proba,pred = self.compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size)
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
        input = input.to(self.device)
        cls = torch.reshape(cls, (-1,))
        cls = cls.type(torch.LongTensor)
        cls = cls.to(self.device)
        logits_g = self.model(input)
        loss_func = nn.CrossEntropyLoss(reduction='none')#weight=self.class_weights
        loss_g = loss_func(logits_g, cls)
        return loss_g.mean(), torch.max(F.softmax(logits_g.detach(),dim=1),1)[0], \
               torch.max(F.softmax(logits_g.detach(),dim=1),1)[1]


a = Ann(batch_size=50,epochs= 8,lr=0.0005,l2=0.05,save_model=False,load_model=False)
#%%
y_pred, y_proba = a.main()

#y_true = pd.read_csv(args.val)
#print(f1_score(y_true['id'], l, average='micro'))
