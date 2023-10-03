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
Simple ResNet model for image regreesion, 
output should be 3D delta (x,y,z) serves as next action prediction.
using cropped/raw img
'''

import numpy as np
seed = 45
torch.manual_seed(seed)

# # 1. Start a W&B Run
# import wandb
# wandb.init(
#     project="amiga-IL-resnet-crop-regression",
#     config={
#     "learning_rate": 0.0001,
#     # "epochs": 10,
#     "batch_size": 16
#     }
# )

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 100

# Define model: ResNet-18
class ImageRegressionResNet(nn.Module):
    def __init__(self, num_regression_output):
        super(ImageRegressionResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_regression_output)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

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
        
        # wandb.log({"loss":  epoch_loss, "mae":  epoch_mae, "rmse":  epoch_rmse})

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}, RMSE: {epoch_rmse:.4f}")
    

# Define dataset (Input: images)
class CustomDataset(Dataset):
    def __init__(self, image_folder, targets_file, transform=None, target_scale_factor=1.0):
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
    def __init__(self, image_folder, targets_file, transform=None, target_scale_factor=1.0):
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
model = ImageRegressionResNet(num_regression_output)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train the model and visualize progress
train_and_visualize(model, train_loader, criterion, optimizer, num_epochs)

# Save the trained model
torch.save(model.state_dict(), 'reg_resnet.pth')


# # Test the model:
# # Load target values from the text file
# targets_test = load_targets_from_file('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/test/targets.txt')
# targets_test = torch.tensor(targets_test, dtype=torch.float32)

# # Scale test target values
# target_scale_factor = 1
# targets_scaled_test = targets_test * target_scale_factor

# # Normalize test targets using the same min-max values as in training
# target_min_test = targets_scaled_test.min(0, keepdim=True)[0]
# target_max_test = targets_scaled_test.max(0, keepdim=True)[0]
# targets_normalized = (targets_scaled_test - target_min_test) / (target_max_test - target_min_test)
# targets_normalized_test = (targets_scaled_test - target_min_test) / (target_max_test - target_min_test)

# # Load the trained model
# # model.load_state_dict(torch.load('image_regression_resnet.pth'))
# model.load_state_dict(torch.load('C:/D/Imperial/Thesis/amiga_dataset/IL/image_regression_resnet.pth'))
# model.eval()

# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# image_test_folder = ('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/test/')
# image_test_files = os.listdir(image_test_folder)
# image_test_files = [file for file in image_test_files if file.endswith('.png')]
# image_paths_test = [os.path.join(image_test_folder, filename) for filename in image_test_files]

# test_dataset = CustomDataset(image_paths_test, targets_test, test_transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# with torch.no_grad():
#     for batch_images, batch_targets in test_loader:
#         predictions = model(batch_images)
#         # Denormalize and descale predictions
#         denormalized_predictions = (predictions * (target_max_test - target_min_test)) + target_min_test
#         denormalized_predictions /= target_scale_factor
#         # Visualize predicted values and ground truth
#         for i in range(denormalized_predictions.size(0)):
#             print(f"Predicted: {denormalized_predictions[i].cpu().numpy()}, Ground Truth: {targets_test[i].cpu().numpy()}")
