#Create a pytorch custom dataset with samples in each file of the directory and the label is the name of the file 
#In each file, each sample is separated by ; and each sample is composed of several list separated by a line
#The first list is the label of the sample and the others are the features of the sample
#The features are separated by a line





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import random 
from model import Net
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

#MyDataset takes a bool argument to know if it is a train dataset or a test dataset

#Each sample must be a tensor 
#Each label must be a string

bestaccuracy = 0 

class MyDataset(Dataset):
    def __init__(self, root_dir, train):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.files.sort()
        self.samples = []
        self.labels = []
        sample = []
        framenb = 0 
        labelchange = {"slideleft": 0, "slideright" : 1, "slideup" : 2, "slidedown" : 3, "longtouch" : 4}
        for file in self.files:
            if file.endswith('.txt'):
            
                with open(os.path.join(root_dir, file), 'r') as f:
                    for line in f:
                        
              
                        if line != '\n' :
                            print("len sample : ",len(sample), "framenb : ", framenb)
                            if line != ';\n':

                                lineprocessed = line.split(',')
                                lineprocessed[0] = lineprocessed[0].replace('[', '')
                                lineprocessed[-1] = lineprocessed[-1].replace(']\n', '')
                                lineprocessed = [i.replace(' ', '') for i in lineprocessed]
                                lineprocessed = [float(i) for i in lineprocessed]
                                sample.append([lineprocessed[x:x+11] for x in range(0, len(lineprocessed),11)])
                                
                               
                                framenb += 1
                            

              
                        #lineprocessed = [float(i) for i in lineprocessed]
                      
                            if line == ";\n"  or framenb == 5 :
                                if framenb == 5 : 
                                    self.samples.append(sample)
                             
                                    self.labels.append(labelchange[str(file)[:-4]])
                                sample = []
                                framenb = 0
                          
                
                        
                
        #If train = True we take RANDOMELY 80% of the dataset for the train dataset and the rest for the test dataset
        #Take randomly 80% of a list 
        if train:
            self.samples = [self.samples[i] for i in sorted(random.sample(range(len(self.samples)), int(len(self.samples)*0.8)))]
            self.labels = [self.labels[i] for i in sorted(random.sample(range(len(self.labels)), int(len(self.labels)*0.8)))]
        else:
            self.samples = [self.samples[i] for i in sorted(random.sample(range(len(self.samples)), int(len(self.samples)*0.2)))]
            self.labels = [self.labels[i] for i in sorted(random.sample(range(len(self.labels)), int(len(self.labels)*0.2)))]
        



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
            #print(torch.Tensor(sample).size())
        return torch.Tensor(sample), label








#The output is a tensor of 5 float (the probability of each class)
#Create train and test method using criterion CorssEntropyLoss
#The optimizer is Adam with a learning rate of 0.001
#The number of epoch is 10
#The batch size is 4
#The number of workers is 4
#The drop_last is True
#The model is saved in the file model.pt
#The accuracy is printed at the end of the training


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        logits = model(data)
        #print("logits : ", logits, "target : ", target)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        predictions = torch.softmax(logits, dim=1).argmax(dim=1)
        loss.backward()

        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    global bestaccuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print("output : ", output, "target : ", target)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    if accuracy > bestaccuracy :
        bestaccuracy = accuracy 
        torch.save(model.state_dict(), 'model_weighs.pth')
        print("New best accuracy : ", bestaccuracy)










def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MyDataset("/home/hugo/Bureau/ARTICPROJECT/CNNMove/test/MucaMoveDataset/dataset ",True)
    test_dataset = MyDataset("/home/hugo/Bureau/ARTICPROJECT/CNNMove/test/MucaMoveDataset/dataset ",False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
    for i,v in enumerate(train_loader):
        print("train : ", v[1])
    for i,v in enumerate(test_loader):
        print("test: ", v[1])
    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #use tqdm to display the progress bar
    for epoch in (range(100)):
        train(model, device, train_loader, optimizer, epoch,criterion)
        test(model, device, test_loader, criterion)



if __name__ == '__main__':
 #open a file to read
    # with open("/Users/hugo/ArcticProject/CNNMOVE/MucaMoveDataset/dataset /slideright.txt") as f:
    #     for line in f : 
    #         print(type(line))
    main()