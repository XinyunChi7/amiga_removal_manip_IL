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
from torchvision.transforms import functional as F
import cv2
import torchvision.transforms as T

'''
(FOR MULTIPLE IMAGES)
This script is used to only show the areas correspond to masks and change other areas to black.
CROPPED with a box centered at the lowest point in mask[1]: whisk.

Will be further applied as inputs for state classification model.

If you want to visualize the masks in gray/color, 
check 'gene_mask_vis_test.py'
'''

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

root = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/all_data/img'
img_paths = [os.path.join(root, filename) for filename in sorted(os.listdir(root))]

# Load and preprocess images
transforms = get_transform(train=False)  
images = []
for img_path in img_paths:
    img = Image.open(img_path).convert("RGB")
    img = transforms(img)
    images.append(img)

print("Number of imported images:", len(images))
print("Check: shape of image tensor:", images[0].shape)


 
# Visualize predicted output with masks
def visualize_masks(image, masks, index):
    # image_np = image.permute(1, 2, 0).byte().cpu().numpy()
    image_np = np.array(image)
    # print("image_np.shape:", image_np.shape)
    merged_masked_image = np.zeros_like(image_np)
    # print("merged_masked_image.shape:", merged_masked_image.shape)

    for i, mask in enumerate(masks):
        mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=0)
        mask_rgb = np.squeeze(mask_rgb) # reshape masks
        # print("mask_rgb.shape:", mask_rgb.shape)
        masked_image = image_np * mask_rgb  
        # print("masked_image.shape:", masked_image.shape)
        merged_masked_image += masked_image # merge masks on black img

    # plt.imshow(np.transpose(merged_masked_image, (1, 2, 0)))
    # plt.title("Masked Output")
    # plt.axis('off')
    # plt.show()

    # Crop (centered at the lowest point in mask[1]: whisk/strainer)
    lowest_point = np.argmax(masks[1])
    # print("lowest_point:", lowest_point)
    x = lowest_point // image_np.shape[2]
    y = lowest_point % image_np.shape[2]
    print("lowest x (center):", x)
    print("lowest y (center):", y)

    box_size = 500
    if x - box_size // 2 - 75 < 0:
        x = box_size // 2 + 75

    cropped_image = merged_masked_image[
        :, x - box_size // 2 - 75 : x + box_size // 2 - 75  ,
        y - box_size // 2 : y + box_size // 2 
    ]
    cropped_image = np.transpose(cropped_image, (1, 2, 0))
    print("cropped_image.shape:", cropped_image.shape)
    
    output_folder = "C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/all_data/crop_0409/"
    plt.imshow(cropped_image)
    # plt.imshow(cropped_image.astype(np.uint8))
    plt.axis('off')
    # plt.savefig(output_folder, bbox_inches='tight', pad_inches=0)
    # plt.title("Cropped Image")
    # plt.show()

    os.makedirs(output_folder, exist_ok=True)

    # cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    output_filename = f"cropped_image_{index}.png"
    output_path = os.path.join(output_folder, output_filename)
    # cv2.imwrite(output_path, (cropped_image_rgb * 255).astype(np.uint8))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # plt.title("Cropped Image")
    # plt.show()

    print("Saved cropped image", index)

# loop all images
for i, img in enumerate(images):
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        prediction = model([img.to(device)])

    masks = prediction[0]['masks'].cpu().numpy()
    
    # image_np = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().cpu().numpy())
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_np)
    # plt.title("Input Image")

    # plt.subplot(1, 2, 2)
    # segmented_mask = np.sum(masks[0], axis=0) + np.sum(masks[1], axis=0) 
    # plt.imshow(segmented_mask, cmap="gray")
    # plt.title("Segmented Mask")

    # plt.tight_layout()
    # plt.show()

    print("i =", i)

    visualize_masks(img, masks, i)



# # Alternative: Single image test:
# # pick one image from the data set
# img = images[3]

# if torch.cuda.is_available():
#     model.cuda() # CHECK the model/weights... all data should be on GPU, convert using '.cuda()' if needed
    
# with torch.no_grad():
#     prediction = model([img.to(device)])

# # Get the masks from the prediction
# masks = prediction[0]['masks'].cpu().numpy()
# print(masks.shape)

# # Visualize the predicted output with masks
# visualize_masks(img, masks , 99999)