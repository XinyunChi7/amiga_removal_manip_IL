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
import cv2

'''
NOT used

(FOR SINGLE IMAGE: original trial)
This script is used to only show the areas correspond to masks and change other areas to black.

If you want to visualize the masks in gray/color, 
check 'gene_mask_vis_test.py'
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
dataset = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset', get_transform(train=True))
dataset_test = CusDataset('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset', get_transform(train=False))
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# define validation data loaders
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 
# initialize and define the model
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

def visualize_masks(image, masks):
    # image_np = image.permute(1, 2, 0).byte().cpu().numpy()
    image_np = np.array(image)
    print("image_np.shape:", image_np.shape)
    merged_masked_image = np.zeros_like(image_np)
    print("merged_masked_image.shape:", merged_masked_image.shape)

    for i, mask in enumerate(masks):
        mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=0)
        mask_rgb = np.squeeze(mask_rgb) # reshape masks
        # print("mask_rgb.shape:", mask_rgb.shape)
        masked_image = image_np * mask_rgb  
        # print("masked_image.shape:", masked_image.shape)
        merged_masked_image += masked_image # merge masks on black img

    plt.imshow(np.transpose(merged_masked_image, (1, 2, 0)))
    # plt.imshow(merged_masked_image.astype(np.uint8))
    plt.title("Masked Output")
    plt.axis('off')
    plt.show()

    # Crop (centered at the lowest point in mask[1]: whisk)
    lowest_point = np.argmax(masks[1])
    print("lowest_point:", lowest_point)
    lowest_row = lowest_point // image_np.shape[2]
    lowest_col = lowest_point % image_np.shape[2]
    box_size = 500
    cropped_image = merged_masked_image[
        :, lowest_row - box_size // 2 : lowest_row + box_size // 2,
        lowest_col - box_size // 2 : lowest_col + box_size // 2
    ]

    cropped_image = np.transpose(cropped_image, (1, 2, 0))
    plt.imshow(cropped_image)
    # plt.imshow(cropped_image.astype(np.uint8))
    plt.title("Cropped Image")
    plt.axis('off')
    plt.show()

    # Save the cropped image
    output_folder = "C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/vis_test/"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "cropped_image.png")
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, (cropped_image_rgb * 255).astype(np.uint8))


# import raw images for test
# pick one image from the data set
img, _ = dataset_test[7]

if torch.cuda.is_available():
    model.cuda() # CHECK the model/weights... all data should be on GPU, convert using '.cuda()' if needed
    
with torch.no_grad():
    prediction = model([img.to(device)])

# Get the masks from the prediction
masks = prediction[0]['masks'].cpu().numpy()
print(masks.shape)

# Visualize the predicted output with masks
visualize_masks(img, masks)