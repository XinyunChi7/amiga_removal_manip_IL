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
import torchvision.transforms as T


'''
(FOR MULTIPLE IMAGES)
This script is used to show only the visualized masks and change background areas to black.
CROPPED with a box centered at RAW IMAGE ITSELF! 

Will be further applied as inputs for regression model.

Check 'test_vis_crop_multi.py' for CROPPED with a box centered at the lowest point in mask[1] (whisk/strainer).
Check 'test_vis_sqcrop_multi.py' for CROPPED with a box centered at the raw image.

'''

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

# Load the trained model from .pth file
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

model_path = "seg_test.pth"
num_classes = 3  
model = get_instance_segmentation_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("Trained model loaded successfully.")


# Import raw images
# to tensor
def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Load list of image paths
root = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/all_data/crop_sq_0409/img'
img_paths = [os.path.join(root, filename) for filename in sorted(os.listdir(root))]
# Load images
transforms = get_transform(train=False)  # Use train=True if needed
images = []
for img_path in img_paths:
    img = Image.open(img_path).convert("RGB")
    img = transforms(img)
    images.append(img)

print("Number of imported images:", len(images))
print("Shape of the first image tensor:", images[0].shape)


# Visualize the predicted output with masks
def visualize_masks(image, masks, index):
    # Create canvas with a black background
    canvas = np.zeros((*masks[0][0].shape, 4), dtype=np.uint8)
    canvas[:, :, 0:3] = 0  # Set the background to black

    # Overlay mask[0] in specific colors   
    canvas[:, :, 0] = masks[0][0] * 255  # Red for mask 0
    canvas[:, :, 2] = masks[1][0] * 255  # Blue for mask 1
    canvas[:, :, 3] = np.maximum(masks[0][0], masks[1][0]) * 128  

    # Crop
    canvas_center_x = canvas.shape[1] // 2
    canvas_center_y = canvas.shape[0] // 2
    box_size = 620
    x_start = max(0, canvas_center_x - box_size // 2)
    x_end = min(canvas.shape[1], canvas_center_x + box_size // 2)
    y_start = max(0, canvas_center_y - box_size // 2)
    y_end = min(canvas.shape[0], canvas_center_y + box_size // 2)

    # crop with a fixed box size (600 by default)
    cropped_canvas = canvas[y_start:y_end, x_start:x_end, :]
    cropped_canvas_rgb = cropped_canvas[:, :, 0:3]

    # Save cropped image
    output_folder = "C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/all_data/test0409/"
    plt.imshow(cropped_canvas_rgb)
    plt.axis('off')  
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"cropped_image_{index}.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    

# loop all images
for i, img in enumerate(images):
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        prediction = model([img.to(device)])

    masks = prediction[0]['masks'].cpu().numpy()
    print("i =", i)

    visualize_masks(img, masks, i)


# # Single image test
# # pick one image from the data set
# img = images[5]

# if torch.cuda.is_available():
#     model.cuda() # CHECK the model/weights... all data should be on GPU, convert using '.cuda()' if needed
    
# with torch.no_grad():
#     prediction = model([img.to(device)])

# # Get the masks from the prediction
# masks = prediction[0]['masks'].cpu().numpy()
# print(masks.shape)

# # Visualize the predicted output with masks
# visualize_masks(img, masks , 3)