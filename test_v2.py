import os
import cv2
import tqdm
import numpy as np

class XimeaCamera():
    def __init__(self, 
                 roi=(0.18, 0.02, 0.75, 0.85) #[x1, y1, x2, y2]
                ):
        import sys
        sys.path.insert(0, "/home/ICT2000/rli/package/api/Python/v3")
        from ximea import xiapi
        from ctypes import c_uint, c_float, c_void_p, c_double

        self.bbox = np.ceil(np.array(roi) * 2048)
        print (self.bbox)
        #create instance for first connected camera
        cam = xiapi.Camera()

        #start communication
        #to open specific device, use:
        #cam.open_device_by_SN('41305651')
        #(open by serial number)
        print('Opening first camera...')
        cam.open_device()
        cam.enable_auto_wb()

        #settings
        cam.set_exposure(100000)
        cam.set_imgdataformat("XI_RGB24")
        print('Exposure was set to %i us' %cam.get_exposure())
        print (cam.get_downsampling_type())

        
        

        cam.set_width(512)
        cam.set_height(512)
        cam.set_offsetX(256)
        cam.set_offsetY(32)

        # cam.set_downsampling("XI_DWN_4x4")

        #create instance of Image to store image data and metadata
        img = xiapi.Image()

        #start data acquisition
        print('Starting data acquisition...')
        cam.start_acquisition()

        self.cam = cam
        self.img = img

    def get_image(self):
        #get data and pass them from camera to img
        self.cam.get_image(self.img)
        rgb = self.img.get_image_data_numpy()
        return rgb

    def close(self):
        #stop data acquisition
        print('Stopping acquisition...')
        self.cam.stop_acquisition()

        #stop communication
        self.cam.close_device()

if __name__ == "__main__":
    camera = XimeaCamera()
    
    for i in tqdm.tqdm(range(1000)):
        rgb = camera.get_image()
    # cv2.imwrite("ximea.png", rgb)
    
    camera.close()
