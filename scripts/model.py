import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

def train_model(model,criterion,optimizer,train_loader,n_epochs,device):
    """
    Trains the neural network on the training data. Iterates based on the 
    number of epochs specified. Will output the loss to graph using matplotlib
    if needed
    
    """
    
    loss_over_time = [] 
    
    model = model.to(device) # Send model to GPU if available
    model.train() # Set the model to training mode
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            
            # Get the input images and labels, and send to GPU if available
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the weight gradients
            optimizer.zero_grad()

            # Forward pass to get outputs
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagation to get the gradients with respect to each weight
            loss.backward()

            # Update the weights
            optimizer.step()

            # Convert loss into a scalar and add it to running_loss
            running_loss += loss.item()
            
            if i % 100 == 99:    
                avg_loss = running_loss / 100
                loss_over_time.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {:.4f}'.format(epoch + 1, i+1, avg_loss))
                running_loss = 0.0

    return loss_over_time

def test_model(model,test_loader,device):
    """
    Tests the neural network on the test data. Will output the 
    accuracy and recall of the trained model
    
    """
    
    # Turn autograd off
    with torch.no_grad():

        # Set the model to evaluation mode
        model = model.to(device)
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []

        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(inputs)
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits,dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(),axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            y_true.extend(labels.cpu().numpy())

        # Calculate the accuracy
        test_preds = np.array(test_preds)
        y_true = np.array(y_true)
        test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
        
        # Recall for each class
        recall_vals = []
        for i in range(10):
            class_idx = np.argwhere(y_true==i)
            total = len(class_idx)
            correct = np.sum(test_preds[class_idx]==i)
            recall = correct / total
            recall_vals.append(recall)
    
    return test_acc,recall_vals

if __name__ == '__main__':
    
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the data
    train_data = torchvision.datasets.ImageFolder(root=str(os.getcwd()) + '/data/processed/train/', transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=str(os.getcwd()) + '/data/processed/test/', transform=transform)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)


    # Define the model
    model = resnet152(pretrained=True)

    # Number of out units is number of classes 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_data.classes))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Move the model to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 10
    
    cost_path = train_model(model,criterion,optimizer,train_loader,num_epochs,device)
    
    # Calculate the test set accuracy and recall for each class
    acc,recall_vals = test_model(model,test_loader,device)
    print('Test set accuracy is {:.3f}'.format(acc))

    # Save the full model
    model_dir = 'models/'
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    filename = 'resnet152.pt'
    torch.save(model, model_dir+filename)