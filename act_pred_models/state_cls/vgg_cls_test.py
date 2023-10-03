import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torchvision.models as models
'''
Trained VGG models (VGG19) is tested on test set for state classification task,
the training can be found in 'vgg_cls_train.py'
'''

# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = 5
model = models.vgg19(pretrained=True).to(device)
model.classifier[6] = nn.Linear(4096, num_classes).to(device)
model.load_state_dict(torch.load('vgg19_cls.pth'))
model.to(device)
model.eval()  

# Define data transforms (ensure they match the ones used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
# Load the test dataset
test_dataset = ImageFolder(root='C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls_no_bg/testing', transform=transform) 
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Create empty lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Define class names 
class_names = ["good", "too_high", "too_left", "too_low", "too_right"]  

# Test the model
def test_model(model, val_loader, num_classes):
    with torch.no_grad():
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct / total

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        confusion = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))

        print('Accuracy of the model on the test images: {} %'.format(100 *acc))

    return confusion

confusion_matrix = test_model(model, test_loader, num_classes)

# Visualize the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(np.arange(num_classes) + 0.5, labels=range(num_classes))
plt.yticks(np.arange(num_classes) + 0.5, labels=range(num_classes))
plt.show()

