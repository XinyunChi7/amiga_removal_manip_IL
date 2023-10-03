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
This script is applied to test the trained segmentation model perforamnce on the test dataset.
with visualized output masks. 
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
dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/')
# dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer/')
print(dataset[0]) # check format
 

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
dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset', get_transform(train=False))
dataset_test = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset', get_transform(train=False))
# dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer', get_transform(train=False))
# dataset_test = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer', get_transform(train=False))
 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])



# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

# initialize and define the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

# Test the model
# Load the trained model from .pth file
model_path = "seg_test.pth"
# model_path = "seg_strainer.pth"
num_classes = 3  
model = get_instance_segmentation_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("Trained model loaded successfully.")

# pick one image from the data set
img, _ = dataset[1]

if torch.cuda.is_available():
    model.cuda() # CHECK the model/weights... all data should be on GPU, convert using '.cuda()' if needed
    
with torch.no_grad():
    prediction = model([img.to(device)])

# Get the masks from the prediction
masks = prediction[0]['masks'].cpu().numpy()
# print("----------------------------")
# print("prediction:", prediction)
# print("----------------------------")
# print("masks:", masks)

# Visualize input image and the predicted masks
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
num_masks_to_visualize = 2 
mask_info = [] 

for i in range(num_masks_to_visualize):
    mask = masks[i]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(mask[0], cmap="gray")  # Display one mask
    plt.title(f"Mask {i+1} (1: hook, 2: whisk)")
    plt.axis("off")
    
    # Save mask info to list for JSON
    mask_dict = {
        "mask_id": i,
        "mask_array": mask.tolist()  # Convert to list for JSON serialization
    }
    mask_info.append(mask_dict)
    
    plt.show()

# # Save mask information as a JSON file
# json_filename = "mask_info.json"
# with open(json_filename, "w") as json_file:
#     json.dump(mask_info, json_file)

# print(f"JSON file saved: {json_filename}")


# # Pytorch visualize output turorial:
# obj1_output = prediction[0]
# obj1_masks = obj1_output['masks']
# print("----------------------------")
# print(f"shape = {obj1_masks.shape}, dtype = {obj1_masks.dtype}, "
#       f"min = {obj1_masks.min()}, max = {obj1_masks.max()}")
# # weights = 
# print("For the first obj, the following instances were detected:")
# print([weights.meta["categories"][label] for label in obj1_output['labels']])

# proba_threshold = 0.5
# dog1_bool_masks = obj1_output['masks'] > proba_threshold
# print(f"shape = {dog1_bool_masks.shape}, dtype = {dog1_bool_masks.dtype}")

# # There's an extra dimension (1) to the masks. We need to remove it
# dog1_bool_masks = dog1_bool_masks.squeeze(1)
# from torchvision.utils import draw_segmentation_masks
# cv2.imshow(draw_segmentation_masks(dog1_int, dog1_bool_masks, alpha=0.9))


