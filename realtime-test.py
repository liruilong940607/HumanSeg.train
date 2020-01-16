import sys
import os
import cv2
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ICT2000/rli/package/api/Python/v3")
from ximea import xiapi
import albumentations as albu
import segmentation_models_pytorch as smp

import utils

class XimeaCamera():
    def __init__(self):
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.enable_auto_wb()

        self.cam.set_exposure(50000)
        self.cam.set_imgdataformat("XI_RGB24")
    
        self.img = xiapi.Image()

        print('Starting data acquisition...')
        self.cam.start_acquisition()

    def get_image(self):
        # return 2048x2048x3
        self.cam.get_image(self.img)
        rgb = self.img.get_image_data_numpy()
        return rgb

    def close(self):
        print('Stopping acquisition...')
        self.cam.stop_acquisition()
        self.cam.close_device()


class HumanSeg():
    def __init__(self, ckpt, device="cuda:0"):
        model = smp.Unet(
            'resnet18', 
            encoder_weights='imagenet', 
            classes=2, 
            # activation='softmax'
        ).to(device)

        if ckpt is not None:
            if os.path.exists(ckpt):
                print (f"load ckpt: {ckpt}")
                model.load_state_dict(torch.load(ckpt))
            else:
                print (f"warning! ckpt {ckpt} not found")
        
        self.device = device
        self.model = model


    def __call__(self, input):
        # 2048x2048x3: [0, 255]
        if isinstance(input, np.ndarray):
            return self.seg_demo(input)

        # 1x3x512x512: [-1, 1]
        if isinstance(input, torch.Tensor):
            return self.seg_training(input)

    def seg_demo(self, image):
        # Input: 2048x2048x3, [0, 255]
        # Output: 256x256, [0, 1]
        mat = utils.aug_matrix(
            (1536-369), (1741-41), 
            176, 256, 
            angle_range=(-0, 0), 
            scale_range=(1.0, 1.0), 
            trans_range=(-0.0, 0.0)
        )
        mat = mat.dot(
            np.array([[1.0, 0, -369],
                      [0, 1.0, -41],
                      [0, 0,   1.0]])
        )

        image = cv2.warpAffine(image, mat, (192, 256), borderValue=(127.5, 127.5, 127.5))
        
        image = albu.RandomBrightnessContrast(
            brightness_limit=(0.2, 0.2), 
            contrast_limit=(0.2, 0.2), 
            p=1
        )(image=image)["image"]

        input = np.float32(image)
        input = (input / 255.0 - 0.5) / 0.5 # TO [-1.0, 1.0]
        input = input.transpose(2, 0, 1) # TO [3 x H x W]

        input = torch.from_numpy(input).float()
        input = input.unsqueeze(0).to(self.device)

        output = self.model(input)
        output = F.softmax(output, dim=1)[0, 1] #[256, 192]
        
        # TO numpy
        segmentation = output.data.cpu().numpy()

        window = np.hstack([
            image, 
            cv2.cvtColor(np.uint8(segmentation*255.0), cv2.COLOR_GRAY2BGR)
        ])
        window = cv2.resize(window, (0, 0), fx=3, fy=3)
        cv2.imshow('window',window)

        return segmentation

    def seg_training(self, input):
        pass


if __name__ == "__main__":
    camera = XimeaCamera()
    app = HumanSeg(
        ckpt="./data/snapshots/latest.pt", 
        device="cuda:1"
    )

    try:        
        for _ in tqdm.tqdm(range(100_000_000)):
            # 2048x2048x3: [0, 255]
            image = camera.get_image()
            prediction = app(image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        camera.close()
