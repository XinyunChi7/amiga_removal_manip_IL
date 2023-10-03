import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
# import os
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
A Conv-LSTM-MLP model for image regression, which outputs 3d delta (x,y,z) for next action prediction.
WITH targets normalization/scaleing
WITHOUT set timestep
'''

# Define model
class ImageRegressionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_mlp_hidden_units, num_regression_output):
        super(ImageRegressionLSTM, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm_layers = nn.LSTM(128 * 28 * 28, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, num_mlp_hidden_units),
            nn.ReLU(),
            nn.Linear(num_mlp_hidden_units, num_mlp_hidden_units),
            nn.ReLU()
        )
        self.fc = nn.Linear(num_mlp_hidden_units, num_regression_output)  # Output layer for regression

    def forward(self, x):
        x = self.conv_layers(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1, channels * height * width)
        lstm_out, _ = self.lstm_layers(x)
        
        # Take the last output of LSTM and pass through MLP
        lstm_last_output = lstm_out[:, -1, :]
        mlp_output = self.mlp(lstm_last_output)
        
        output = self.fc(mlp_output)
        return output

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

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 100

# Load target values from txt file
targets_train = read_targets_from_file('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/train/targets.txt')
targets_train = torch.tensor(targets_train, dtype=torch.float32)

# Scale target values
target_scale_factor = 100 
targets_scaled = targets_train * target_scale_factor
# Normalize targets to the range [0, 1]
target_min = targets_scaled.min(0, keepdim=True)[0]
target_max = targets_scaled.max(0, keepdim=True)[0]
targets_normalized = (targets_scaled - target_min) / (target_max - target_min)

# Initialize model
num_regression_output = 3  # Number of regression output values (x, y, z)
input_size = 3 
hidden_size = 128
num_layers = 2
num_mlp_hidden_units = 256
model = ImageRegressionLSTM(input_size, hidden_size, num_layers, num_mlp_hidden_units, num_regression_output)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
image_folder = ('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/train/')
image_files = os.listdir(image_folder)
# Filter out non-image files
image_files = [file for file in image_files if file.endswith('.png')]
image_paths_train = [os.path.join(image_folder, filename) for filename in image_files]

train_dataset = CustomDataset(image_paths_train, targets_train, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Define matrices: Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
def mean_absolute_error(preds, targets):
    absolute_errors = torch.abs(preds - targets)
    return torch.mean(absolute_errors)
def root_mean_squared_error(preds, targets):
    squared_errors = torch.pow(preds - targets, 2)
    mean_squared_error = torch.mean(squared_errors)
    return torch.sqrt(mean_squared_error)



# Test the model:
# Load target values 
targets_test = read_targets_from_file('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/test/targets.txt')
targets_test = torch.tensor(targets_test, dtype=torch.float32)

# Scale & normalize
targets_scaled_test = targets_test * target_scale_factor
targets_normalized_test = (targets_scaled_test - target_min) / (target_max - target_min)

# Load the trained model 
model.load_state_dict(torch.load('C:/D/Imperial/Thesis/amiga_dataset/IL/conv_lstm_mlp_reg.pth'))
model.eval() # evaulate mode

# Load data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_test_folder = ('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset_crop/test/')
image_test_files = os.listdir(image_test_folder)
image_test_files = [file for file in image_test_files if file.endswith('.png')]
image_paths_test = [os.path.join(image_test_folder, filename) for filename in image_test_files]

test_dataset = CustomDataset(image_paths_test, targets_test, test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for batch_images, batch_targets in test_loader:
        predictions = model(batch_images)
        # Denormalize and descale predictions
        denormalized_predictions = (predictions * (target_max - target_min)) + target_min
        denormalized_predictions /= target_scale_factor
        # Visualize predicted values and ground truth
        for i in range(denormalized_predictions.size(0)):
            print(f"Predicted: {denormalized_predictions[i].cpu().numpy()}, Ground Truth: {targets_test[i].cpu().numpy()}")


# evaluate
total_loss = 0.0
total_mae = 0.0
num_samples = 0

with torch.no_grad():
    for batch_images, batch_targets in test_loader:
        predictions = model(batch_images)
        
        loss = criterion(predictions, batch_targets)
        mae = torch.abs(predictions - batch_targets).mean()
        
        # total loss and MAE
        total_loss += loss.item()
        total_mae += mae.item()
        num_samples += batch_images.size(0)

# average loss and MAE
average_loss = total_loss / num_samples
average_mae = total_mae / num_samples

print(f"Average Loss on Test Set: {average_loss:.4f}")
print(f"Average MAE on Test Set: {average_mae:.6f}")


# # OPTIONAL: Visualize predicted values and ground truth
# # trajectory visualization
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# data = list(zip(denormalized_predictions[-10:-2].cpu().numpy(), targets_test[-10:-2].cpu().numpy()))

# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

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

# # Set labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.legend()
# plt.show()
