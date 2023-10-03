import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
Simple resnet model for img regression test,
WITHOUT targets normalization/scaleing'''

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
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
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
num_epochs = 50

# Load target values from the text file
targets_train = read_targets_from_file('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_vismask/train/targets.txt')
# Convert targets to a tensor
targets_train = torch.tensor(targets_train, dtype=torch.float32)
# Normalize targets to the range [0, 1]
target_min = targets_train.min(0, keepdim=True)[0]
target_max = targets_train.max(0, keepdim=True)[0]
targets_normalized = (targets_train - target_min) / (target_max - target_min)

# Init
num_regression_output = 3  # Number of regression output values (x, y, z)
model = ImageRegressionResNet(num_regression_output)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
image_folder = ('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_vismask/train/')
image_files = os.listdir(image_folder)
# Filter out non-image files
image_files = [file for file in image_files if file.endswith('.png')]
image_paths_train = [os.path.join(image_folder, filename) for filename in image_files]

train_dataset = CustomDataset(image_paths_train, targets_train, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# Define a function to calculate Mean Absolute Error (MAE)
def mean_absolute_error(preds, targets):
    absolute_errors = torch.abs(preds - targets)
    return torch.mean(absolute_errors)

# Define a function to calculate Root Mean Squared Error (RMSE)
def root_mean_squared_error(preds, targets):
    squared_errors = torch.pow(preds - targets, 2)
    mean_squared_error = torch.mean(squared_errors)
    return torch.sqrt(mean_squared_error)

# Define a function to train the model and visualize training progress
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}, RMSE: {epoch_rmse:.4f}")
    
    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(maes, label='Mean Absolute Error')
    plt.plot(rmses, label='Root Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss, MAE, and RMSE over Epochs')
    plt.legend()
    plt.show()

# Train the model and visualize progress
train_and_visualize(model, train_loader, criterion, optimizer, num_epochs)

# Save the trained model
torch.save(model.state_dict(), 'image_regression_resnet.pth')



# Test the model:
# import test set
# Load target values from the text file
targets_test = read_targets_from_file('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_vismask/test/targets.txt')
targets_test = torch.tensor(targets_test, dtype=torch.float32)
targets_normalized = (targets_test - target_min) / (target_max - target_min)
# load images
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_test_folder = ('C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_vismask/test/')
image_test_files = os.listdir(image_folder)
image_test_files = [file for file in image_files if file.endswith('.png')]
image_paths_test = [os.path.join(image_folder, filename) for filename in image_files]

test_dataset = CustomDataset(image_paths_test, targets_test, test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load the trained model
model.load_state_dict(torch.load('C:/D/Imperial/Thesis/amiga_dataset/IL/image_regression_resnet.pth'))
model.eval()

with torch.no_grad():
    for batch_images, batch_targets in test_loader:
        predictions = model(batch_images)
        # Visualize predicted values and ground truth
        for i in range(predictions.size(0)):
            print(f"Predicted: {predictions[i].cpu().numpy()}, Ground Truth: {batch_targets[i].cpu().numpy()}")
