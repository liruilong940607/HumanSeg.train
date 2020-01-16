import glob
import tqdm
import cv2
import os
import numpy as np

import utils

## ---------- processing tif to png ----------
# pattern = "./data/alignment/*.tif"

# files = glob.glob(pattern)
# for f in tqdm.tqdm(files):
#     img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
#     img = cv2.resize(img, (1024, 1024))
#     cv2.imwrite(f.replace(".tif", ".jpg"), img)


## ----------- calculate bbox ------------
# pattern = "./data/alignment/*-removebg-preview.png"

# files = glob.glob(pattern)
# x1, y1, x2, y2 = 0, 0, 0, 0
# for f in tqdm.tqdm(files):
#     img = cv2.imread(f, cv2.IMREAD_UNCHANGED)[:, :, 3]
#     h, w = img.shape
#     ys, xs = (img>128).nonzero()
    
#     x1 += (xs.min() - 1) / len(files) / w
#     y1 += (ys.min() - 1) / len(files) / h
#     x2 += (xs.max() + 1) / len(files) / w
#     y2 += (ys.max() + 1) / len(files) / h

# # x1, y1, x2, y2: (0.27707692307692305, 0.10969230769230769, 0.66, 0.8344615384615383)
# print (f"x1, y1, x2, y2: {x1, y1, x2, y2}")


## --------- crop as test image ---------
# x1, y1, x2, y2 = (0.18, 0.02, 0.75, 0.85)

# pattern = "./data/alignment/*.tif"
# os.makedirs("./data/aligned_test_images/", exist_ok=True)

# files = glob.glob(pattern)
# for f in tqdm.tqdm(files):
#     img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
#     height, width, _ = img.shape

#     img = img[int(y1*height):int(y2*height), int(x1*width):int(x2*width)]
#     height, width, _ = img.shape

#     dst_height = 512
#     dst_width = int(dst_height / height * width) 
#     img_resize = cv2.resize(img, (dst_width, dst_height))

#     cv2.imwrite(
#         f.replace(".tif", ".jpg").replace("alignment", "aligned_test_images"), 
#         img_resize
#     )


## -------- align training set -----------
x1, y1, x2, y2 = (0.18, 0.02, 0.75, 0.85)

train_img = "../PIFu-RealTime/test_data_example/sythetic_sample_1/000_00.color.png"
templ_img = "./data/alignment/000001-removebg-preview.png"

train_img = cv2.imread(train_img, cv2.IMREAD_UNCHANGED)[:, :, 3] > 128
templ_img = cv2.imread(templ_img, cv2.IMREAD_UNCHANGED)[:, :, 3] > 128
templ_img = templ_img | templ_img[:, ::-1]

train_ys, train_xs = train_img.nonzero()
templ_ys, templ_xs = templ_img.nonzero()

train_bbox = [
    train_xs.min()/train_img.shape[1], 
    train_ys.min()/train_img.shape[0], 
    train_xs.max()/train_img.shape[1], 
    train_ys.max()/train_img.shape[0],
]
templ_bbox = [
    templ_xs.min()/templ_img.shape[1], 
    templ_ys.min()/templ_img.shape[0], 
    templ_xs.max()/templ_img.shape[1], 
    templ_ys.max()/templ_img.shape[0], 
]

print (train_bbox) # [0.380859375, 0.18359375, 0.630859375, 0.873046875]
print (templ_bbox) # [0.372, 0.13, 0.626, 0.83]

scale_h = (y2 - y1) / (templ_bbox[3] - templ_bbox[1])
scale_w = (x2 - x1) / (templ_bbox[2] - templ_bbox[0])

height = train_bbox[3] - train_bbox[1]
width = train_bbox[2] - train_bbox[0]
center_h = (train_bbox[3] + train_bbox[1]) / 2
center_w = (train_bbox[2] + train_bbox[0]) / 2

bbox = [
    center_w - width * scale_w / 2,
    center_h - height * scale_h / 2,
    center_w + width * scale_w / 2,
    center_h + height * scale_h / 2,
]

print ("final bbox:", bbox)

# final:
# [0.225, 0.120, 0.786, 0.937]

train_img = "../PIFu-RealTime/test_data_example/sythetic_sample_1/000_00.color.png"
img = cv2.imread(train_img, cv2.IMREAD_UNCHANGED)
height, width, _  = img.shape
img = img[int(height*0.120):int(height*0.937), int(width*0.225):int(width*0.786)]

cv2.imwrite("test_alignment.jpg", img)

