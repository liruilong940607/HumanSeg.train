import torch

import segmentation_models_pytorch as smp
from models.BiSeNet import BiSeNet

if __name__ == "__main__":
    import time
    def speed(model, name, height=224, width=224, channel=3):
        with torch.no_grad():
            input = torch.rand(1, channel, height, width).cuda()
            model(input)

            t1 = time.time()
            for i in range(1000):
                # input = torch.rand(1, channel, height, width).cuda()
                output = model(input)
                # print (output.shape)
            torch.cuda.synchronize()
            t2 = time.time()
            print('%10s : %f' % (name, (t2 - t1)/1000))

    # model = BiSeNet(n_class=2, useUpsample=True, useDeconvGroup=True).cuda()
    # torch.cuda.synchronize()
    # speed(model, "BiSeNet_224")
    # torch.cuda.synchronize()    

    # model = BiSeNet(n_class=2, useUpsample=True, useDeconvGroup=True).cuda()
    # torch.cuda.synchronize()
    # speed(model, "BiSeNet_448", height=448, width=448)
    # torch.cuda.synchronize()

    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=2).cuda()
    torch.cuda.synchronize()
    speed(model, "smp_Unet", height=256, width=256)
    torch.cuda.synchronize()
