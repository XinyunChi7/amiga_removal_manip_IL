import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
Test the Resnet18 model for regression
WITH targets normalization/scaleing
'''

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
    
# Define dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        target = torch.tensor(self.targets[idx-1], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# import target values from a text file (x,y,z)
def read_targets_from_file(file_path):
    targets = []
    with open(file_path, 'r') as f:
        for line in f:
            a, b, c = map(float, line.strip().split(','))
            targets.append([a, b, c])
    return targets


batch_size = 16
num_regression_output = 3  # Number of regression output values (x, y, z)
model = ImageRegressionResNet(num_regression_output)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define matrices: MAE, RMSE
def mean_absolute_error(preds, targets):
    absolute_errors = torch.abs(preds - targets)
    return torch.mean(absolute_errors)
def root_mean_squared_error(preds, targets):
    squared_errors = torch.pow(preds - targets, 2)
    mean_squared_error = torch.mean(squared_errors)
    return torch.sqrt(mean_squared_error)


# Test the model:
# Load target values from the text file
targets_test = read_targets_from_file('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_removebadimg/test/targets.txt')
targets_test = torch.tensor(targets_test, dtype=torch.float32)

target_scale_factor = 1  #scale? (no need if there is normalization)
targets_scaled = targets_test * target_scale_factor
# Normalize targets to the range [0, 1] # diff ranges have been tested
target_min = targets_scaled.min(0, keepdim=True)[0]
target_max = targets_scaled.max(0, keepdim=True)[0]
# targets_normalized = (targets_scaled - target_min) / (target_max - target_min)
# Normalize the targets to the range (-10, 10)
targets_normalized = 20 * (targets_test - target_min) / (target_max - target_min) - 10
print("targets_normalized:", targets_normalized)

# Load the trained model
# model.load_state_dict(torch.load('image_regression_resnet.pth'))
model.load_state_dict(torch.load('C:/D/Imperial/Thesis/amiga_dataset/IL/reg_resnet.pth'))
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_test_folder = ('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_removebadimg/test/')
image_test_files = os.listdir(image_test_folder)
image_test_files = [file for file in image_test_files if file.endswith('.png')]
image_paths_test = [os.path.join(image_test_folder, filename) for filename in image_test_files]

test_dataset = CustomDataset(image_paths_test, targets_test, test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

mae_total = 0.0
rmse_total = 0.0
num_samples = 0
with torch.no_grad():
    for batch_images, batch_targets in test_loader:
        predictions = model(batch_images)
        # Denormalize and descale predictions
        # denormalized_predictions = (predictions * (target_max - target_min)) + target_min
        denormalized_predictions = ((predictions + 10) / 20) * (target_max - target_min) + target_min
        denormalized_predictions /= target_scale_factor
        # Visualize predicted values and ground truth
        for i in range(denormalized_predictions.size(0)):
            print(f"Predicted: {denormalized_predictions[i].cpu().numpy()}, Ground Truth: {targets_test[i].cpu().numpy()}")
        
        batch_mae = mean_absolute_error(predictions, batch_targets).item()
        batch_rmse = np.sqrt(root_mean_squared_error(predictions, batch_targets)).item()

    mae_total += batch_mae
    rmse_total += batch_rmse
    num_samples += batch_size

average_mae = mae_total / num_samples
average_rmse = rmse_total / num_samples
print("Average MAE:", average_mae)
print("Average RMSE:", average_rmse)


# # trajectory visualization
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Separate denormalized_predictions and targets_test points
# data = list(zip(denormalized_predictions[-10:-2].cpu().numpy(), targets_test[-10:-2].cpu().numpy()))

# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Separate the points into x, y, and z components
# x_pred = [point[0][0] for point in data]
# y_pred = [point[0][1] for point in data]
# z_pred = [point[0][2] for point in data]

# x_ground = [point[1][0] for point in data]
# y_ground = [-point[1][1] for point in data]
# z_ground = [point[1][2] for point in data]

# # Plot predicted points in blue
# ax.scatter(x_pred, y_pred, z_pred, c='blue', marker='o', label='Predicted')

# # Plot ground truth points in red
# ax.scatter(x_ground, y_ground, z_ground, c='red', marker='x', label='Ground Truth')

# # Set labels for the axes
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()
# plt.show()
