import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import wandb
wandb.init(
    project="amiga-IL-cls-retrain",
    config={
    "learning_rate": 0.001,
    "batch_size": 8
    }
)

'''
ResNet models (ResNet18/50) has been trained and tested for state classification task,
with poor performance in losses
'''
# Data source: C:\D\Imperial\Thesis\amiga_dataset\cv_eef_dataset\imgs_test\whisk_with_hook\IL_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 75
batch_size = 16
learning_rate = 1e-3

# Data augmentation (OPTIONAL)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# Import data
train_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls_no_bg/training'
val_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls_no_bg/testing'
dataset_train = datasets.ImageFolder(train_data_path, transform)
print(dataset_train.class_to_idx)
# Test set
# dataset_test = datasets.ImageFolder('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing', transform_test)
dataset_test = datasets.ImageFolder(val_data_path, transform)
print(dataset_test.class_to_idx)

# Load data
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# Define the model
# model = models.resnet18().to(device)
model = models.resnet50().to(device)

# Define Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Update learning rate
# def update_lr(optimizer, lr):
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr
def update_lr(optimizer, epoch): # decay lr every 50 epochs
    curr_lr = learning_rate * (0.1 ** (epoch // 50))
    print("lr:", curr_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = curr_lr


# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
sum_loss = 0
correct_predictions = 0
total_predictions = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print_loss = loss.data.item()
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

        print('epoch:{},loss:{}'.format(epoch, ave_loss))

    # Decay learning rate
    if (epoch + 1) % 20 == 0:
       curr_lr /= 3
       update_lr(optimizer, curr_lr)



# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = correct / total

    print('Accuracy of the model on the test images: {} %'.format(100 *acc))



