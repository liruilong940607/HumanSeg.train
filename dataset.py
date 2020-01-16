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
        albu.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=1),
        albu.RandomBrightness(limit=0.2, p=1),
        albu.RandomContrast(limit=0.4, p=1),

        albu.JpegCompression(quality_lower=70, quality_upper=100, p=1),
        albu.Blur(blur_limit=7, p=1),

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

        image_names = [f for f in os.listdir(image_dir) if f[-3:]=="jpg"]
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

            extra_image_files = sorted(glob.glob("/home/ICT2000/rli/local/data/Removebg2/images/*.jpeg"))
            extra_label_files = [f.replace("/images/", "/labels/").replace(".jpeg", "-removebg-preview.png") for f in extra_image_files]
            image_files += extra_image_files
            label_files += extra_label_files

            # extra_image_files = sorted(glob.glob("/home/ICT2000/rli/local/data/Supervisely/images/*.jpeg"))
            # extra_label_files = [f.replace("/images/", "/labels/").replace(".jpeg", ".png") for f in extra_image_files]
            # image_files += extra_image_files
            # label_files += extra_label_files

            # extra_image_files = sorted(glob.glob("/home/ICT2000/rli/local/data/Supervisely/images/*.png"))
            # extra_label_files = [f.replace("/images/", "/labels/").replace(".png", ".png") for f in extra_image_files]
            # image_files += extra_image_files
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
        if label.shape[2] == 3:
            label = np.float32(label > 0) * 255

        label = cv2.resize(label, (width, height)) 
        
        if self.train:
            image1 = self.augmentation(image=image)["image"]
            image2 = self.augmentation(image=image)["image"]

            ys, xs = (label[:, :, -1] > 128).nonzero()
            width = xs.max() - xs.min()
            height = ys.max() - ys.min()

            mat = utils.aug_matrix(
                width, height, 
                self.input_size, self.input_size, 
                angle_range=(-20, 20), 
                scale_range=(0.7, 1.1), 
                trans_range=(-0.3, 0.3)
            )
            mat = mat.dot(
                np.array([[1.0, 0, -xs.min()],
                          [0, 1.0, -ys.min()],
                          [0, 0,   1.0]])
            )

            
            if random.random() < 0.5:
                image1 = np.flip(image1, 1)
                image2 = np.flip(image2, 1)
                label = np.flip(label, 1)

            image1 = cv2.warpAffine(image1, mat, (self.input_size, self.input_size), borderValue=(127.5, 127.5, 127.5))
            image2 = cv2.warpAffine(image2, mat, (self.input_size, self.input_size), borderValue=(127.5, 127.5, 127.5))
            label = cv2.warpAffine(label, mat, (self.input_size, self.input_size))     

            image1 = np.float32(image1)
            image1 = (image1 / 255.0 - 0.5) / 0.5 # TO [-1.0, 1.0]
            image1 = image1.transpose(2, 0, 1) # TO [3 x H x W]

            image2 = np.float32(image2)
            image2 = (image2 / 255.0 - 0.5) / 0.5 # TO [-1.0, 1.0]
            image2 = image2.transpose(2, 0, 1) # TO [3 x H x W]

            mask = label[:, :, -1] > 128

            # to tensor
            image1 = torch.from_numpy(image1).float()
            image2 = torch.from_numpy(image2).float()
            mask = torch.from_numpy(mask).long()
            
            return image1, image2, mask


        else:
            # for 1024
            mat = utils.aug_matrix(
                (1536-369)/2, (1741-41)/2, 
                176, 256, 
                angle_range=(-0, 0), 
                scale_range=(1.0, 1.0), 
                trans_range=(-0.0, 0.0)
            )
            mat = mat.dot(
                np.array([[1.0, 0, -369/2],
                          [0, 1.0, -41/2],
                          [0, 0,   1.0]])
            )

            image = cv2.warpAffine(image, mat, (192, 256), borderValue=(127.5, 127.5, 127.5))
            label = cv2.warpAffine(label, mat, (192, 256))     

            image = np.float32(image)
            image = (image / 255.0 - 0.5) / 0.5 # TO [-1.0, 1.0]
            image = image.transpose(2, 0, 1) # TO [3 x H x W]

            mask = label[:, :, -1] > 128

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

    dataset = Dataset(
        input_size=256, 
        image_dir="./data/alignment", 
        label_dir="./data/alignment",
        train=False,
    )

    images = []
    for i in range(13):
        image, mask = dataset[i]
        images.append(image)
    images = torch.stack(images)

    input_norm = images * 0.5 + 0.5 #[-1, 1] -> [0, 1]
    torchvision.utils.save_image(
        input_norm,
        f"./image.jpg", 
        normalize=True, range=(0, 1), nrow=4, padding=10, pad_value=0.5
    )


    