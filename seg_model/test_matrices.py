import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from torchvision.transforms import functional as F
import cv2
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

'''
Test seg model performance on test set with confusion matrix: 

Expected output:
true_positives, true_negatives, false_positives, false_negatives
93 0 0 7
Total Test Samples: 50
Accuracy: 100.00%
Precision: 100.00%
Recall: 93.00%
F1-score: 96.37%
Mean IoU: 81.93%
'''

# Define dataset
class CusDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "cv2_mask"))))
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "cv2_mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
 
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        # print(obj_ids)
 
        masks = mask == obj_ids[:, None, None]
 
        # object detection > bounding box
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)
 
# Define model
def get_instance_segmentation_model(num_classes):
    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # define mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model

# Define data transform
def get_transform(train):
    transforms = []
    # converts image to Tensor!
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# load dataset
dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset', get_transform(train=True))
dataset_test = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset', get_transform(train=False))
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# load data
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 
# init
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)


# Test the model
# Load the trained model from .pth file
model_path = "seg_test.pth"
num_classes = 3  
model = get_instance_segmentation_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("Trained model loaded successfully.")


# Compute IoU 
threshold = 0.4
def calculate_iou(predicted_mask, target_mask):
    # Convert the masks to binary (0, 1)
    predicted_binary = (predicted_mask > threshold).astype(np.uint8)
    target_binary = (target_mask > threshold).astype(np.uint8)
    # print("predicted_binary", predicted_binary)
    # print("target_binary", target_binary)

    # Calculate iou
    intersection = np.logical_and(predicted_binary, target_binary)
    union = np.logical_or(predicted_binary, target_binary)
    # print("intersection", intersection)
    # print("union", union)
    iou = np.sum(intersection) / np.sum(union)

    return iou


# Init
total_samples = 50  # number of test samples 
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

true_labels = []
predicted_labels = []
iou_values = []

# Define the IoU threshold and label accuracy threshold
iou_threshold = 0.4
label_accuracy_threshold = True  

# Loop through the test dataset for the specified number of samples
for idx, (image, target) in enumerate(test_loader):
    if idx >= total_samples:
        break
    with torch.no_grad():
        prediction = model(image)
    
    for i in range(len(target[0]['masks'])):
        # Calculate IoU 
        predicted_mask = prediction[0]['masks'][i].squeeze().cpu().numpy()
        target_mask = target[0]['masks'][i].squeeze().cpu().numpy()
        IoU = calculate_iou(predicted_mask, target_mask) 
        
        # Check label accuracy 
        predicted_label = prediction[0]['labels'].cpu().numpy()[i]
        target_label = target[0]['labels'].cpu().numpy()[i]
        label_accuracy = (predicted_label == target_label)
        
        # Store
        true_labels.append(target_label)
        predicted_labels.append(predicted_label)
        iou_values.append(IoU)
        
        # Update confusion matrix values
        if IoU > iou_threshold and label_accuracy >= label_accuracy_threshold:
            if target_label == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if target_label == 1:
                false_negatives += 1
            else:
                false_positives += 1

# Calculate precision, recall, and F1-score
print("true_positives, true_negatives, false_positives, false_negatives")
print(true_positives, true_negatives, false_positives, false_negatives)

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate mean IoU
mean_iou = np.mean(iou_values)

# Print the performance metrics
print(f"Total Test Samples: {total_samples}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print(f"Mean IoU: {mean_iou * 100:.2f}%")



# Plot confusion matrix
confusion_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

# Plot the confusion matrix using ConfusionMatrixDisplay
class_names = ["Background", "Mask"]
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names)

# Plot cm
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Segmentation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

