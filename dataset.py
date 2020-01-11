import os
import glob
import random 

import cv2
import numpy as np
import torch

import albumentations as albu

import utils

def get_training_augmentation():
    train_transform = [
        # albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=1),
        albu.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
        albu.RandomBrightness(limit=0.2, p=1),
        albu.RandomContrast(limit=0.4, p=1),

        # albu.JpegCompression(quality_lower=70, quality_upper=100, p=1),
        albu.Blur(blur_limit=5, p=1),

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)

class Dataset(object):
    def __init__(self, 
                 input_size=256, 
                 image_dir="./data/images", 
                 label_dir="./data/labels",
                 train=True,
                 ):
        super().__init__()
        self.input_size = input_size
        self.train = train

        image_names = os.listdir(image_dir)
        image_files = [os.path.join(image_dir, f) for f in image_names]
        label_files = [
            os.path.join(
                label_dir, 
                f.replace(".jpg", "-removebg-preview.png")
            ) for f in image_names
        ]

        #extra images
        if train:
            extra_image_files = sorted(glob.glob("/home/ICT2000/rli/local/data/Removebg/images/*.jpeg"))
            extra_label_files = [f.replace("/images/", "/labels/").replace(".jpeg", "-removebg-preview.png") for f in extra_image_files]
            image_files += extra_image_files
            label_files += extra_label_files

        self.image_files = []
        self.label_files = []
        for image_file, label_file in zip(image_files, label_files):
            if os.path.exists(image_file) and os.path.exists(label_file):
                self.image_files.append(image_file)
                self.label_files.append(label_file)

        self.image_files = self.image_files
        self.label_files = self.label_files
        
        self.augmentation = get_training_augmentation()
            
        print (f"Dataset: {self.__len__()}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label_file = self.label_files[index]

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # TO RGB
        height, width, _ = image.shape

        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label, (width, height)) 

        ys, xs = (label[:, :, 3] > 128).nonzero()
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
    
        if self.train:
            mat = utils.aug_matrix(
                width, height, 
                self.input_size, self.input_size, 
                angle_range=(-0, 0), 
                # scale_range=(0.7, 0.7), 
                scale_range=(0.35, 0.75), 
                trans_range=(-0.4, 0.4),
                # trans_range=(-0.0, 0.0)
            )
            mat = mat.dot(
                np.array([[1.0, 0, -xs.min()],
                          [0, 1.0, -ys.min()],
                          [0, 0,   1.0]])
            )

            image = self.augmentation(image=image)["image"]
            if random.random() < 0.5:
                image = np.flip(image, 1)
                label = np.flip(label, 1)

        else:
            mat = utils.aug_matrix(
                width, height, 
                self.input_size, self.input_size, 
                angle_range=(-0, 0), 
                scale_range=(0.6, 0.6), 
                trans_range=(-0.0, 0.0)
            )
            mat = mat.dot(
                np.array([[1.0, 0, -xs.min()],
                          [0, 1.0, -ys.min()],
                          [0, 0,   1.0]])
            )

        image = cv2.warpAffine(image, mat, (self.input_size, self.input_size))
        label = cv2.warpAffine(label, mat, (self.input_size, self.input_size))        

        image = (image / 255.0 - 0.5) / 0.5 # TO [-1.0, 1.0]
        image = image.transpose(2, 0, 1) # TO [3 x H x W]
        mask = label[:, :, 3] > 128

        # to tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask

if __name__ == "__main__":
    import torchvision

    dataset = Dataset(
        input_size=256, 
        image_dir="/home/ICT2000/rli/local/data/LIP/ATR/humanparsing/JPEGImages/", 
        label_dir="/home/ICT2000/rli/local/data/LIP/ATR/humanparsing/RemoveBG/",
    )

    # dataset = Dataset(
    #     input_size=256, 
    #     image_dir="./data/test_images", 
    #     label_dir="./data/test_labels",
    # )

    images = []
    for i in range(16):
        image, mask = dataset[i]
        images.append(image)
    images = torch.stack(images)

    input_norm = images * 0.5 + 0.5 #[-1, 1] -> [0, 1]
    torchvision.utils.save_image(
        input_norm,
        f"./image.jpg", 
        normalize=True, range=(0, 1), nrow=4, padding=10, pad_value=0.5
    )


    