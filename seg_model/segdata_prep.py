import base64
import json
import os
import os.path as osp
import PIL.Image
from labelme import utils
import random

'''
RUN this script to:

0. images in 'img' folder and annotations in 'mask' folder are required.
NOTE: the names of images in 'img' folder might be needed to be renamed for simplification,
      If so, please refer to 'img_rename.py.'

1. convert labelme json to cv2_mask images for visualization, 
which will be further fed into segmentation model with original images for training.
NOTE: the cv2_mask images are NOT binary images, different masks are represented by different colors.
      If binary images are needed, please refer to 'mask_vis_save.py.'

2. split the dataset into train and val sets, 
   and save the names of images (train set / val set) in txt files.
'''

def main():
 
    jsonpath = r"C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer/mask_strainer"
    outpath = r"C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer/cv2_mask"
    ratio = 0.8

    if not osp.exists(outpath):
        os.mkdir(outpath)
 
    namelist=[]
    paths = os.listdir(jsonpath)
    for onepath in paths:
        name = onepath.split(".")[0]
        json_file = jsonpath+"\\"+onepath
        outfile = outpath+"\\"+name+".png"
        namelist.append(name)
 
        data = json.load(open(json_file))
        imageData = data.get("imageData")
 
        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)
 
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )
        utils.lblsave(outfile, lbl)
 
    random.shuffle(namelist)
    n_total = len(namelist)
    offset = int(n_total * ratio)
    train = namelist[:offset]
    val = namelist[offset:]
 
    with open(outpath+"\\train.txt","w") as f:
        for i in range(len(train)):
            f.write(str(train[i])+"\n")
    f.close()
    with open(outpath+"\\val.txt","w") as f:
        for i in range(len(val)):
            f.write(str(val[i])+"\n")
    f.close()
 
 
if __name__ == "__main__":
    main()