import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os # run this when there's gpu memory error
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
InceptionV3 (GoogleNetV3) is trained for state classification task,
with 5 classes: 0, 1, 2, 3, 4
using cropped image with background and visualized mask edge

NOT very good perforamnce with this dataset!
'''

# imprt data
train_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls'
#val_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls/testing'

batch_size = 16
num_classes = 6

# Data augmentation and normalization
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load the datasets
train_dataset = datasets.ImageFolder(train_data_path, transform=data_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load InceptionV3 (pre-trained)
model = models.inception_v3(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Define loss function + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Check whether GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_loss_values = []
train_accuracy_values = []

# Training loop
num_epochs = 100 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # Calculate training accuracy and loss for the current epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / total_predictions

    train_loss_values.append(epoch_loss)
    train_accuracy_values.append(epoch_accuracy)

    # Print training statistics for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Visualize training process
# Plotting the training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1),  train_loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
save_path = "C:/D/Imperial/Thesis/amiga_dataset/IL/cls_test.pth"
torch.save(model.state_dict(), save_path)
print("Trained model saved successfully.")


# # Test the model
# model.eval()

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in val_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         acc = correct / total

#     print('Accuracy of the model on the test images: {} %'.format(100 *acc))
