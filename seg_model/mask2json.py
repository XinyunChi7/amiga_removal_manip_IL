import cv2
import os
import json
import sys
 
'''
This script can be used to covert mask to json file,
BUT is NOT very accurate!
'''

def func(file:str) -> dict:
    png = cv2.imread(file)
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    dic = {"version": "5.0.1", "flags": {},"shapes":list(), "imagePath":os.path.basename(file),
            "imageHeight":png.shape[0], "imageWidth":png.shape[1]}
    for contour in contours:
        temp = list()
        for point in contour[2:]:
            if len(temp) > 1 and temp[-2][0] * temp[-2][1] * int(point[0][0]) * int(point[0][1]) != 0 and (int(point[0][0]) - temp[-2][0]) * (
                            temp[-1][1] - temp[-2][1]) == (int(point[0][1]) - temp[-2][1]) * (temp[-1][0] - temp[-1][0]):
                temp[-1][0] = int(point[0][0])
                temp[-1][1] = int(point[0][1])
            else:
                temp.append([int(point[0][0]), int(point[0][1])])
        dic["shapes"].append({"label": "result", "points":temp, "group_id": None,
                                "shape_type": "polygon", "flags": {}})
 
    return dic
 
 
if  __name__ == "__main__":
 
    if len(sys.argv) != 3:
        raise ValueError("Usage: python mask2json.py <input_folder> <output_folder>")
 
    if os.path.isdir(sys.argv[1]):
        for file in os.listdir(sys.argv[1]):
            with open(os.path.join(sys.argv[2], os.path.splitext(file)[0]+".json"), mode='w', encoding="utf-8") as f:
                json.dump(func(os.path.join(sys.argv[1], file)), f)
    else:
        with open(os.path.join(sys.argv[2], os.path.splitext(os.path.basename(sys.argv[1]))[0]+".json"), mode='w', encoding="utf-8") as f:
            json.dump(func(sys.argv[1]), f)