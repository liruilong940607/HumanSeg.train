import os
import cv2
import numpy as np
import PIL
import tqdm
import glob

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp

def _transform_matrix(H, w, h):
    H0 = np.concatenate((H, np.array([[0, 0, 1]])), axis=0)
    A = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
    A_inv = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h/ 2.0], [0, 0, 1]])
    H0 = A.dot(H0).dot(A_inv)
    H0 = np.linalg.inv(H0)
    return H0

class HumanSeg():
    def __init__(self, ckpt=None, device="cuda:0"):
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
        

        # (0.18, 0.02, 0.75, 0.85)
        mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
        mask[:, :, 10:435, 92:384] = 1.0
        self.mask = torch.from_numpy(mask).to(device)
        
        ################################3

        # ---- forward ----
        matrix_forward = np.array([
            [0.42905028, 0.0, 22.23910615],
            [0.0, 0.42905028, 22.45363128],
            [0.0, 0.0, 1.0]
        ])
        matrix_forward_tensor = torch.from_numpy(
            _transform_matrix(matrix_forward[0:2], 512, 512)
        ).to(device).unsqueeze(0)
        flow_forward_tensor = F.affine_grid(
            theta=matrix_forward_tensor[:, 0:2], 
            size=(1, 3, 512, 512)
        ).float().to(device)
        self.flow_forward_tensor = flow_forward_tensor[:, :256, :256, :]
        
        # ---- backward ----
        matrix_backward = np.linalg.inv(np.array([
            [0.42905028*2, 0.0, 22.23910615],
            [0.0, 0.42905028*2, 22.45363128],
            [0.0, 0.0, 1.0]
        ]))
        matrix_backward_tensor = torch.from_numpy(
            _transform_matrix(matrix_backward[0:2], 256, 256)
        ).to(device).unsqueeze(0)
        flow_backward_tensor = F.affine_grid(
            theta=matrix_backward_tensor[:, 0:2], 
            size=(1, 2, 256, 256)
        ).float().to(device)
        self.flow_backward_tensor = flow_backward_tensor[:, :256, :256, :]

    def __call__(self, x, save=None):
        batchsize = x.size(0)

        input = F.grid_sample(
            x * 0.5 + 0.5, # TO [0, 1]
            self.flow_forward_tensor.repeat(batchsize, 1, 1, 1), 
            mode='bilinear', 
            padding_mode='zeros'
        ) # 'zeros' | 'border' 
        input = (input - 0.5) / 0.5 # TO [-1, 1]

        output = self.model(input)
        output = F.softmax(output, dim=1)
        
        output = F.grid_sample(
            output, 
            self.flow_backward_tensor.repeat(batchsize, 1, 1, 1), 
            mode='bilinear', 
            padding_mode='zeros'
        ) # 'zeros' | 'border' 
        output = F.interpolate(output, scale_factor=2)
        output = self.mask * output

        if save is not None:
            input_norm = x * 0.5 + 0.5  #[-1, 1] -> [0, 1]
            output_norm = output[:, 1:2].repeat(1, 3, 1, 1).float()
            torchvision.utils.save_image(
                torch.cat([input_norm, output_norm * input_norm], dim=0), save, 
                normalize=True, range=(0, 1), nrow=len(input_norm), padding=10, pad_value=0.5
            )
        
        return output[:, 1, :, :]

def test_single_img():
    # image = cv2.imread("./data/alignment/000301.jpg")
    image = cv2.imread("/home/ICT2000/rli/Downloads/test_epoch13_idx12000_rp_sophia_posed_003.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512)) / 255.0
    image = (image.transpose(2, 0, 1) - 0.5) / 0.5 # in [-1, 1]

    input = torch.from_numpy(image).unsqueeze(0).to("cuda:0").float()

    app = HumanSeg(
        ckpt="./data/snapshots/latest-ckpt.pt", 
        device="cuda:0"
    )
    # prediction is [1, 1, 512, 512] tensor 
    prediction = app(input, save="./test.jpg") # in [0, 1]

def test_video():
    files = sorted(glob.glob("./data/alignment/*.jpg"))
    files = [files[i] for i in [0, 50, 100, 150, 200, 250, 300, 350, 400]]
    
    tensor = []
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512)) / 255.0
        image = (image.transpose(2, 0, 1) - 0.5) / 0.5 # in [-1, 1]

        input = torch.from_numpy(image).unsqueeze(0).to("cuda:0").float()
        tensor.append(input)
    tensor = torch.cat(tensor, dim=0)

    app = HumanSeg(
        ckpt="./data/snapshots/latest-ckpt.pt", 
        device="cuda:0"
    )
    # prediction is [1, 1, 512, 512] tensor 
    prediction = app(tensor, save="./test.jpg") # in [0, 1]

if __name__ == "__main__":
    test_single_img()

    