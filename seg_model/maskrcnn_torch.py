import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from torchvision.transforms import functional as F
import json

'''
Complete training + testing for seg model.

A Mask R-CNN model (backbone: ResNet-50-FPN) is trained on the AMIGA dataset for image segmentation.
NOTE that 2 segmentation models have been pretrained on amiga dataset, for strainer and whisker respectively:
  1. seg_test.pth > whisk (by default)
  2. seg_strainer.pth > strainer
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

# load dataset 
dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer')
print(dataset[0]) # check format: print the first image and its label
 

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
dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer', get_transform(train=True))
dataset_test = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer', get_transform(train=False))
 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

# Load data
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 
# init model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 3 # 2 class (hook+whisk) + background, NOTE TO PLUS 1 
model = get_instance_segmentation_model(num_classes)
model.to(device)
 
# define optimizer and lr
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
 
# training
num_epochs = 75
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update lr
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# Save the trained model
save_path = "C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/seg_strainer.pth"
torch.save(model.state_dict(), save_path)
print("Trained model saved successfully.")


# Test the model
# pick one image from the data set
img, _ = dataset_test[0]
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# Get the masks from the prediction
masks = prediction[0]['masks'].cpu().numpy()
# print("----------------------------")
# print("prediction:", prediction)
# print("----------------------------")
# print("masks:", masks)

# Visualize the input image and the predicted masks
image_np = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().cpu().numpy())
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Input Image")

plt.subplot(1, 2, 2)
segmented_mask = np.sum(masks[0], axis=0) + np.sum(masks[1], axis=0) 
plt.imshow(segmented_mask, cmap="gray")
plt.title("Segmented Mask")

plt.tight_layout()
plt.show()

# Visualize and save masks
num_masks_to_visualize = 2  # Number of masks to visualize and save
mask_info = []  # To store mask information for saving as JSON

for i in range(num_masks_to_visualize):
    mask = masks[i]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(mask[0], cmap="gray")  # Display one mask
    # plt.title(f"Mask {i+1} (1: hook, 2: whisk)")
    plt.title(f"Mask {i+1} (1: hook, 2: strainer)")
    plt.axis("off")
    
    # Save mask info to list for JSON
    mask_dict = {
        "mask_id": i,
        "mask_array": mask.tolist()  # Convert to list for JSON serialization
    }
    mask_info.append(mask_dict)
    
    plt.show()

# # Save mask information as a JSON file (TBM)
# json_filename = "mask_info.json"
# with open(json_filename, "w") as json_file:
#     json.dump(mask_info, json_file)

# print(f"JSON file saved: {json_filename}")
