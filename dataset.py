import os
import glob
import random 

import cv2
import numpy as np
import torch

class Dataset(object):
    def __init__(self, input_size=256, image_dir="./data/images", label_dir="./data/labels"):
        super().__init__()
        self.input_size = input_size

        image_names = os.listdir(image_dir)
        image_files = [os.path.join(image_dir, f) for f in image_names]
        label_files = [
            os.path.join(
                label_dir, 
                f.replace(".jpg", "-removebg-preview.png")
            ) for f in image_names
        ]

        self.image_files = []
        self.label_files = []
        for image_file, label_file in zip(image_files, label_files):
            if os.path.exists(image_file) and os.path.exists(label_file):
                self.image_files.append(image_file)
                self.label_files.append(label_file)

        self.image_files = self.image_files * 10
        self.label_files = self.label_files * 10
        
            
        print (f"Dataset: {self.__len__()}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label_file = self.label_files[index]

        image = cv2.imread(image_file)
        image = cv2.resize(image, (self.input_size, self.input_size))
        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        # cv2.imwrite(label_file, label)
        label = cv2.resize(label, (self.input_size, self.input_size))

        # aug
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # TO RGB
        image = image / 255.0 - 0.5 # TO [-0.5, 0.5]
        image = image.transpose(2, 0, 1) # TO [3 x H x W]
        mask = label[:, :, 3] > 128

        # to tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask

if __name__ == "__main__":
    dataset = Dataset()
    image, mask = dataset[0]
    print(image.shape)
    print(mask.shape)


    