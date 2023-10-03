import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
'''
NOT USED
Conv-LSTM model for image regreesion, 
output should be 3D delta (x,y,z) serves as next action prediction.
Using cropped/raw img
'''

import numpy as np
seed = 45
torch.manual_seed(seed)

# 1. Start a W&B Run
import wandb
wandb.init(
    project="amiga-IL-conv-lstm-regression-crop",
    config={
    "learning_rate": 0.001,
    # "epochs": 10,
    "batch_size": 16
    }
)

# Hyperparameters
batch_size = 16
learning_rate = 0.01
num_epochs = 50

# Define LSTM-based model
import torch
import torch.nn as nn
class ImageRegressionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_regression_output):
        super(ImageRegressionLSTM, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, padding=1),# input channels 3, output channels 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # input channels 32, output channels 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm_layers = nn.LSTM(64 * 56 * 56, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_regression_output)
        # self.fc = nn.Linear(hidden_size, 64)
        # self.fc = nn.Linear(64, num_regression_output)
    
    def forward(self, x):
        x = self.conv_layers(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1, channels * height * width)
        lstm_out, _ = self.lstm_layers(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


# Define function to train the model
def train_and_visualize(model, train_loader, criterion, optimizer, num_epochs):
    losses = []
    maes = []
    rmses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_mae = 0.0
        running_rmse = 0.0
        for batch_images, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_images)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_mae += mean_absolute_error(predictions, batch_targets).item()
            running_rmse += root_mean_squared_error(predictions, batch_targets).item()
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        epoch_mae = running_mae / len(train_loader)
        maes.append(epoch_mae)
        epoch_rmse = running_rmse / len(train_loader)
        rmses.append(epoch_rmse)
        
        wandb.log({"loss":  epoch_loss, "mae":  epoch_mae, "rmse":  epoch_rmse})

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}, RMSE: {epoch_rmse:.4f}")
    

# Define dataset (Input: images)
class CustomDataset(Dataset):
    def __init__(self, image_folder, targets_file, transform=None, target_scale_factor=10000.0):
        self.image_folder = image_folder
        self.targets = torch.tensor(load_targets_from_file(targets_file), dtype=torch.float32)
        self.transform = transform
        self.target_scale_factor = target_scale_factor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_name = f"{idx + 1:03}.png"
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx] * self.target_scale_factor # Scale the target values
        return image, target

class CustomDataset_test(Dataset):
    def __init__(self, image_folder, targets_file, transform=None, target_scale_factor=10000.0):
        self.image_folder = image_folder
        self.targets = torch.tensor(load_targets_from_file(targets_file), dtype=torch.float32)
        self.transform = transform
        self.target_scale_factor = target_scale_factor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_name = f"{idx + 228:03}.png"
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx] * self.target_scale_factor # Scale the target values
        return image, target

# Load targets from a text file (Output: x,y,z)
def load_targets_from_file(targets_file):
    targets = []
    with open(targets_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                image_name = values[0]
                a = float(values[1]) if len(values) > 1 else 0.0
                b = float(values[2]) if len(values) > 2 else 0.0
                c = float(values[3]) if len(values) > 3 else 0.0
                targets.append([a, b, c])
    return targets

# Calculate the mean and standard deviation from original targets
original_targets = load_targets_from_file(targets_file='C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/train/targets.txt')
mean = np.mean(original_targets, axis=0)
std = np.std(original_targets, axis=0)
print("mean:", mean)
print("std:", std)

# Load dataset
train_images_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/train/'
train_targets_file = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/train/targets.txt'
test_images_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/test/'
test_targets_file = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/test/targets.txt'

# reivsed input
final_image_size = (224, 224)  
num_regression_output = 3  # Number of regression output values (x, y, z)

# Data augmentation
transform = transforms.Compose([
    transforms.Resize(final_image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_images_folder, train_targets_file, transform=transform)
test_dataset = CustomDataset_test(test_images_folder, test_targets_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Matrices: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) 
def mean_absolute_error(preds, targets):
    absolute_errors = torch.abs(preds - targets)
    return torch.mean(absolute_errors)

def root_mean_squared_error(preds, targets):
    squared_errors = torch.pow(preds - targets, 2)
    mean_squared_error = torch.mean(squared_errors)
    return torch.sqrt(mean_squared_error)


# training
# Initialize model
num_regression_output = 3  # Number of regression output values (x, y, z)
input_size = 3  
hidden_size = 128
num_layers = 2
num_mlp_hidden_units = 128
model = ImageRegressionLSTM(input_size, hidden_size, num_layers, num_regression_output)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Train the model and visualize progress
train_and_visualize(model, train_loader, criterion, optimizer, num_epochs)

# Save the trained model
torch.save(model.state_dict(), 'conv_lstm_reg.pth')

# test
model.eval()
with torch.no_grad():
    mae_total = 0.0
    rmse_total = 0.0
    num_samples = 0
    for batch_images, batch_targets in test_loader:
        batch_size, num_channels, height, width = batch_images.size()
        batch_images = batch_images.view(batch_size, -1, num_channels * height * width) 
        predictions = model(batch_images)
        batch_mae = mean_absolute_error(predictions, batch_targets).item()
        batch_rmse = root_mean_squared_error(predictions, batch_targets).item()

        # Denormalize the predictions using the calculated mean and std
        denormalized_predictions = predictions * std + mean
        denormalized_targets = batch_targets / 10000.0  # sacle back

        # Visualize predicted values and ground truth
        for i in range(predictions.size(0)):
            # print(f"Predicted: {predictions[i].cpu().numpy()}, Ground Truth: {batch_targets[i].cpu().numpy()}")
            print("Predicted (denormalized):", denormalized_predictions[i])
            print("Ground Truth:", batch_targets[i])

       

