# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

#construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,required=True,
                help="path to trained pytorch model")
args=vars(ap.parse_args())

#set the device we will be using to train the model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the KMNIST dataset and randomly grab 10 data points
print("[INFO] loading KMNIST dataset...")
testData=KMNIST(root="data",train=False,transform=ToTensor(),download=True)
idxs=np.random.randint(0,len(testData),size=(10,))
testData=Subset(testData,idxs)

#initialize the data loader
testDataLoader=DataLoader(testData,batch_size=1)

#load the model from disk
model=torch.load(args["model"]).to(device)

#switch off autograd
with torch.no_grad():
    #loop over the test data
    for (image,label) in testDataLoader:
        #grab the original image and resize it
        origImage=image.numpy().squeeze(axis=(0,1))
        getLabel=testData.dataset.classes[label.numpy()[0]]
        
        #send the input to the device and make the predictions
        image=image.to(device)
        pred=model(image)
        
        #find the class label with the highest probability
        idx=pred.argmax(axis=1).cpu().numpy()[0]
        predLabel=testData.dataset.classes[idx]
        #convert the image from grayscale to RGB(so we can draw on it)
        #and resize it(so we can more easily see it on the screen )
        origImage=np.dstack([origImage]*3)
        origImage=imutils.resize(origImage,width=128)
        
        #draw the class label on the image
        color=(0,255,0) if predLabel==getLabel else (0,0,255)
        cv2.putText(origImage,predLabel,(2,25),cv2.FONT_HERSHEY_SIMPLEX,0.95,color,2)
        
        #display the result in terminal and show the input image
        print("[INFO] Predicted: {}, groung truth or Actual: {}".format(predLabel,getLabel))
        cv2.imshow("Image",origImage)
        cv2.waitKey(0)
        

