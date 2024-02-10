# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from cnn_arch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()

ap.add_argument("-m","--model",type=str,required=True,help="path to output model")

ap.add_argument("-p","--plot",type=str,required=True,help="path to output loss/accuracy plot")

args=vars(ap.parse_args())

#define training hyperparameters

EPOCHS=10

INIT_LR=1e-3

BATCH_SIZE=64

#define the train and test split

TRAIN_SPLIT=0.75
VAL_SPLIT=1-TRAIN_SPLIT

#initialize the device

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the dataset

print("[INFO] loading the dataset...")
train_dataset=KMNIST(root="data",train=True,transform=ToTensor(),download=True)
test_dataset=KMNIST(root="data",train=False,transform=ToTensor(),download=True)

#calculate the train and test split
num_train_samples=int(len(train_dataset)*TRAIN_SPLIT)
num_val_samples=int(len(train_dataset)*VAL_SPLIT)

#split the dataset

train_dataset,val_dataset=random_split(train_dataset,[num_train_samples,num_val_samples],generator=torch.Generator().manual_seed(42))

#initialize the data loaders
trainDataLoader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valDataLoader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
testDataLoader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

#initialize the model
print("[INFO] initializing the model...")
model=LeNet(numChannels=1,classes=len(train_dataset.dataset.classes)).to(DEVICE)


#initialize the optimizer
optimizer=Adam(model.parameters(),lr=INIT_LR)

#initialize the loss function
lossFunc=nn.NLLLoss()

#initialize the dictionary to store training history

H={"train_loss":[],
   "train_acc":[],
   "val_loss":[],
   "val_acc":[]}

#measure how long the training is going to take

startTime=time.time()

#loop over the epochs

for e in range(EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss=0
    totalValLoss=0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect=0
    valCorrect=0
    # loop over the training set
    for (x,y) in trainDataLoader:
        #send the input to the device
        (x,y)=x.to(DEVICE),y.to(DEVICE)
        # perform a forward pass and calculate the training loss
        pred=model(x)
        loss=lossFunc(pred,y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss+=loss
        trainCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()
        # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x,y) in valDataLoader:
            #send the input to the device
            (x,y)=x.to(DEVICE),y.to(DEVICE)
            # make the predictions and calculate the validation loss
            pred=model(x)
            totalValLoss+=lossFunc(pred,y)
            # calculate the number of correct predictions
            valCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()
            
    #calculate the average training and validation loss
    avgTrainLoss=totalTrainLoss/len(trainDataLoader)
    avgValLoss=totalValLoss/len(valDataLoader)
    #calculate the training and validation accuracy
    trainCorrect/=len(train_dataset)
    valCorrect/=len(val_dataset)
    #update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    #print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e+1,EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss,trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss,valCorrect))
    #finish measuring how long training is taking
    endTime=time.time()
    print("[INFO] total time taken: {:.2f}s".format(endTime-startTime))
    #we can now evaluate the network on the test set
    print("[INFO] evaluating network...")
    #turn off autograd for testing evaluation
    with torch.no_grad():
        #set the model in evaluation mode
        model.eval()
        #initialize a list to store our predictions
        preds=[]
        #loop over the test set
        for (x,y) in testDataLoader:
            #send the input to the device
            x,y=x.to(DEVICE),y.to(DEVICE)
            #make the predictions and calculate the loss
            pred=model(x)
            preds.extend(pred.argmax(1).cpu().numpy())
    #generate the classifications report
    print(classification_report(test_dataset.targets.cpu().numpy(),
                                np.array(preds),target_names=test_dataset.classes))
    #plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"],label="train_loss")
    plt.plot(H["val_loss"],label="val_loss")
    plt.plot(H["train_acc"],label="train_acc")
    plt.plot(H["val_acc"],label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    #serialize the model to disk
    torch.save(model,args["model"])
    print("[INFO] training complete...")
        
                
