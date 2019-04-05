import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import random
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchsummary import summary
from tqdm import tqdm_notebook as tqdm
from torch.autograd import Variable

from imfunc import undersample, to_original, centeredCrop, kspace_recon
from load import mri_dataset
import unet

# Dataset, dataloader
train_file_dir = '/home/harry/fastmri/data/pad_train_single'
val_file_dir = '/home/harry/fastmri/data/pad_val_single'

traindata = mri_dataset(train_file_dir)
valdata = mri_dataset(val_file_dir)

batch_size = 1

train_loader = torch.utils.data.DataLoader(dataset=traindata, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=valdata, 
                                           batch_size=batch_size, 
                                           shuffle=True)

# Mount unet on gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet.UNet(2,2)
model.to(device)

print(summary(model, input_size = (2,720,720)))

def train_net(model, epochs = 5, lr = 0.001, save_cp = True, 
              trainloader=None, valloader=None, sample_k=None):
    
    dir_checkpoint = '/home/harry/fastmri/CheckPoint/'
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    
    Train_losses = []
    Val_losses = []
    
    for epoch in tqdm(range(epochs)):
        # Train mode
        
        print("Training: ", end='')
        model.train()
        
        train_epoch_loss = 0
        train_samples = 0
        
        for inputs, gt in trainloader:
            inputs = inputs.to(device=device, dtype=torch.float)
            gt = gt.to(device=device, dtype=torch.float)
            
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output, gt)
            train_epoch_loss += loss.item()
            train_samples += inputs.shape[0] # size of batch
            
            loss.backward()
            optimizer.step()
            if train_samples % 20 == 0:
                print("#", end='')
        
        # Validation mode
        
        print(' end')
        print("Validation: ", end ='')
        model.eval()
        
        val_epoch_loss = 0
        val_samples = 0
        
        for inputs, gt in valloader:
            with torch.no_grad():
                inputs = inputs.to(device=device, dtype=torch.float)
                gt = gt.to(device=device, dtype=torch.float)
                
                output = model(inputs)
                
                loss = criterion(output, gt)
                val_epoch_loss += loss.item()
                val_samples += inputs.shape[0]
                
                if val_samples % 10 == 0:
                    print("#", end='')
        print(' end')
        
        # Check reconstructed img
        # 4-fold
        os.chdir('/home/harry/fastmri/data/pad_val_single')
        sample = np.load(sample_k)
        recon_c, model_recon_c, gt = kspace_recon(sample, 4)
        plt.figure()
        plt.subplot(131)
        plt.imshow(abs(recon_c), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(abs(model_recon_c), cmap='gray')
        plt.title('After U-net')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(abs(gt), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.show()
        
        Train_losses.append(train_epoch_loss/train_samples)
        Val_losses.append(val_epoch_loss/val_samples)
        
        print('Epoch: {}/{}, Train loss: {}, Val loss: {}'.format(epoch+1, epochs,
                                                               train_epoch_loss/train_samples,
                                                               val_epoch_loss/val_samples))
        
        # Save model parameters, losses for every 50 epochs
        if save_cp:
            if (epoch+1) % 50 == 0:
                torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch+1))
                np.save(dir_checkpoint + 'Train_loss{}.npy'.format(epoch+1), Train_losses)
                np.save(dir_checkpoint + 'Validation_loss{}.npy'.format(epoch+1), Val_losses)
                print('Checkpoint {} saved!'.format(epoch + 1))
       
    return model, Train_losses, Val_losses


sample_k = 'pad_val_array9.npy'
model, Train_losses, Val_losses = train_net(model, trainloader=train_loader,
                                           valloader=val_loader, sample_k=sample_k)

# Plot train loss & validation loss
train_loss = np.array(Train_losses)
val_loss = np.array(Val_losses)
plt.figure()
x = list(range(100))
plt.plot(x,train_loss, x, val_loss)
plt.title('lr = 1e-3')
plt.xlabel('epochs')
plt.legend(('Train loss','Val loss'))
plt.show()