import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
VGG models (VGG19) has been trained and tested for state classification task
'''

import wandb
wandb.init(
    project="amiga-IL-cls-retrain",
    config={
    "learning_rate": 0.001,
    "batch_size": 8
    }
)

# Data source: C:\D\Imperial\Thesis\amiga_dataset\cv_eef_dataset\imgs_test\whisk_with_hook\IL_test

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 16
learning_rate = 1e-3

# Data augmentation (OPTIONAL)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
train_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls_no_bg/training'
val_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls_no_bg/testing'
dataset_train = datasets.ImageFolder(train_data_path, transform)
dataset_test = datasets.ImageFolder(val_data_path, transform)
print(dataset_train.class_to_idx)

# Load data
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# Define the model
num_classes = 5
model = models.vgg19(pretrained=True).to(device)
model.classifier[6] = nn.Linear(4096, num_classes).to(device)

# Define Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Train the model
total_step = len(train_loader)
correct_predictions = 0
total_predictions = 0

for epoch in range(num_epochs):
    model.train()  
    sum_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print_loss = loss.item()
        sum_loss += print_loss
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    ave_loss = sum_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / total_predictions
    wandb.log({"loss":  ave_loss, "accuracy": epoch_accuracy  })

    print('Epoch:{}, Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, ave_loss, epoch_accuracy))
    scheduler.step()

# Save trained model
torch.save(model.state_dict(), 'vgg19_cls_test.pth')


# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         acc = correct / total

#     print('Accuracy of the model on the test images: {} %'.format(100 *acc))

