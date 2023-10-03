import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os # run this when there's gpu memory error
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
'''
InceptionV3 (GoogleNetV3) is trained for state classification task,
with 5 classes: 0, 1, 2, 3, 4
using cropped image with no background and visualized mask

For model training, please check 'inceptionv3_cls_crop_no_bg_train.py'
'''


# import data
val_data_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/cls_dataset/crop_cls_no_bg/testing'

batch_size = 16
num_classes = 5

# Data augmentation and normalization
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the datasets
val_dataset = datasets.ImageFolder(val_data_path, transform=data_transform)
# Create data loaders
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Load InceptionV3 (pretrained)
num_classes = 5
model = models.inception_v3(pretrained=True)
# Replace the fully connected layer to match num_classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Check whether GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the saved model state_dict
model_path = "cls_test.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
print("Trained model loaded successfully.")

print(model)
# print image shape for each layer
for name, layer in model.named_children():
    print(name)
    print(layer)
    print('----------------------')



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

confusion_matrix = test_model(model, val_loader, num_classes)

# Visualize the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(np.arange(num_classes) + 0.5, labels=range(num_classes))
plt.yticks(np.arange(num_classes) + 0.5, labels=range(num_classes))
plt.show()

