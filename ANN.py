import torch, collections, datetime, ast
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from PIL import Image, ImageFile
from images_classes import *
import cv2 as cv
from aug import *


############# [136, -14, 241, 116] 142641

class DataArgs:
    def __init__(self, csv_path='csv/',batch_size = 50,epochs = 50,lr = 1e-3,l2=0.01,img_size=500,hidden=None,
                 save_model=None,load_model=None):

        self.csv_path = csv_path
        self.train = 'train.csv'
        self.val = 'validation.csv'
        self.embedding = 'embedding'
        self.target = 'name'
        self.num_features= 512
        self.hidden = hidden
        self.in_channel = 1
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_model = save_model
        self.load_model = load_model
        self.l2 = l2
        self.img_size = img_size


class FRBSDataset(Dataset):
    def __init__(self, values,train_set=True):  # train=False
        self.values = values
        self.train_set = train_set

    def __getitem__(self, index): # 0: 'path', 1: 'cls', 2: 'face_indexes', 3 'aug'

        image_class = torch.tensor(self.values[index][1])
        image = ImageInSet(a.train_dl.dataset.values[index][0])
        image = image.values.crop(image.get_face_indexes())
        #indices = Image.open(self.values[index][0]).get_face_indexes()
        #image = ImageInSet(self.values[index][0])
        #indices = image.get_face_indexes()
        #image = ImageInSet(self.values[index][0])
        #face_image=image.get_face_image(eval(self.values[index][2])).values
        #indices = eval(self.values[index][2])
        #image = cv.imread(self.values[index][0])
        #face_image = image[indices[1]:indices[3],indices[0]:indices[2]]
        #print(index)
        if self.train_set and self.values[index][3]:
            face_image = aug_img(image)
        else:
            face_image = aug_img2(image)
        norm_image = cv.normalize(face_image, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        resized_img = torch.tensor(norm_image)
        resized_img = resized_img.permute(2, 0, 1)
        return resized_img, image_class

    def __len__(self):
        return len(self.values)



class Ann:
    def __init__(self, args):
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.train_dl = None
        self.val_dl = None
        self.input_features = None
        self.out_features = None
        #self.class_weights = None
        self.num_workers = 0
        self.init_dls()
        self.model = InceptionResnetV1(classify=True,pretrained='vggface2', num_classes=self.out_features)
        self.optimizer = self.optimizer()
        self.training_loss = 0.0
        self.val_loss = 0.0
        self.all_training_loss = list()
        self.all_val_loss = list()
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.y_pred = list()
        self.y_pred_proba = list()


    #def model(self):
     #   model = Net(self.input_features,args.hidden,self.out_features)
      #  if self.use_cuda:
       #     print(f"Using CUDA; {torch.cuda.device_count()} devices.")
        #    model = model.to(self.device)
        #return model

    def optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
        #return optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.99,weight_decay=self.args.l2)

    def lr_schedule(self,epoch):
      if (epoch + 1) % 8 == 0:
        self.args.lr *= 0.96
        print(self.args.lr)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.99,weight_decay=self.args.l2)

    def init_dls(self):
        temp_pd = pd.read_csv(self.args.train)
        #self.input_features = len(ast.literal_eval(temp_pd.iloc[0,2]))
        self.out_features = len(temp_pd['class'].value_counts())
        train_dataset = FRBSDataset(values=temp_pd.values,)
        self.train_dl = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,
                                       num_workers=self.num_workers,pin_memory=self.use_cuda)
        #sorted_dict = collections.OrderedDict(sorted(temp_pd['class'].value_counts().to_dict().items()))
        #self.class_weights = torch.FloatTensor(1 - (np.array(list(sorted_dict.values())) / sum(sorted_dict.values())))
        temp_pd = pd.read_csv(self.args.val)
        val_dataset = FRBSDataset(values=temp_pd.values,train_set=False)
        self.val_dl = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,
                                     num_workers=self.num_workers,pin_memory=self.use_cuda)

    def main(self,decay_learning=False):
        if self.args.load_model is not False:
            try:
                self.model.load_state_dict(torch.load(self.args.load_model))
                print('Loaded model for training')
            except FileNotFoundError:
                print('File Not Found')
                return
        else:
            print('Start training')
        for epoch_ndx in range(self.args.epochs):
            print(f"epoch: {epoch_ndx}")
            if decay_learning:
                self.lr_schedule(epoch_ndx)
            trn_loss = self.training(epoch_ndx, self.train_dl)
            self.all_training_loss.append(trn_loss.detach() / len(self.train_dl))
            val_loss = self.validation(epoch_ndx, self.val_dl)
            self.all_val_loss.append(val_loss.detach() / len(self.val_dl))
            print(f"trn: {trn_loss / len(self.train_dl):.3f}, val: {val_loss / len(self.val_dl):.3f}")
            print(f"{(trn_loss / len(self.train_dl)) - (val_loss / len(self.val_dl)):.3f}")
        plt.figure()
        plt.plot(self.all_training_loss, 'r', label="Train")
        plt.plot(self.all_val_loss, 'b', label="Validation")
        plt.suptitle("Model: {self.time} lr: {self.args.lr} hidden layer: {self.args.hidden_layer}")
        plt.legend(loc="upper right")
        plt.xlabel('epochs')  # fontsize=18
        plt.ylabel('loss')
        plt.show()
        if args.save_model:
            torch.save(self.model.state_dict(), f'{self.time}_model.pth')
        print("Finished")
        return self.y_pred, self.y_pred_proba

    def training(self, epoch_ndx, train_dl):
        self.model.train()
        self.training_loss = 0.0
        for batch_ndx, batch_tup in enumerate(train_dl, 0):
            self.optimizer.zero_grad()
            loss, _,_ = self.compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size)
            self.training_loss += loss
            loss.backward()
            self.optimizer.step()
        return self.training_loss

    def validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            self.val_loss = 0.0
            for batch_ndx, batch_tup in enumerate(val_dl, 0):
                loss, proba,pred = self.compute_batch_loss(batch_ndx, batch_tup, val_dl.batch_size)
                self.val_loss += loss
                if (epoch_ndx + 1) % self.args.epochs == 0:
                    for i in pred:
                        self.y_pred.append(int(i))
                    for i in proba:
                        self.y_pred_proba.append(round(float(i), 2))
        return self.val_loss

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size):
        input, cls = batch_tup
        cls = torch.reshape(cls, (-1,))
        cls = cls.type(torch.LongTensor)
        logits_g = self.model(input)
        loss_func = nn.CrossEntropyLoss(reduction='none')#weight=self.class_weights
        loss_g = loss_func(logits_g, cls)
        return loss_g.mean(), torch.max(F.softmax(logits_g.detach(),dim=1),1)[0], \
               torch.max(F.softmax(logits_g.detach(),dim=1),1)[1]

args = DataArgs(batch_size= 20,epochs= 20,hidden=350,lr=0.0003,l2=0.01,save_model=False,load_model=False)
a = Ann(args)
#%%

l,t = a.main()
y_true = pd.read_csv(args.val)
print(f1_score(y_true['id'], l, average='micro'))
