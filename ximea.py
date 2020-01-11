import sys 
import cv2
sys.path.insert(0, "/home/ICT2000/rli/package/api/Python/v3")
import ximea
from ximea import xiapi
from tqdm import tqdm

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
print(cam.get_downsampling())
cam.set_imgdataformat("XI_RGB24")
# cam.set_width(512)
# cam.set_height(512)
print('Exposure was set to %i us' %cam.get_exposure())

#create instance of Image to store image data and metadata
img = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

# for i in tqdm(range(1000)):
while True:
    #get data and pass them from camera to img
    cam.get_image(img)

    #get raw data from camera
    #for Python2.x function returns string
    #for Python3.x function returns bytes
    rgb = img.get_image_data_numpy()
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # nparr = np.fromstring(data_raw, np.uint8).reshape
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imshow("demo", data_raw)

    #transform data to list
    # data = list(data_raw)
    # Our operations on the frame come here
    # rgb = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow('frame',rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # #print image data and metadata
    # print('Image number: ' + str(i))
    # print('Image width (pixels):  ' + str(img.width))
    # print('Image height (pixels): ' + str(img.height))
    # print('First 10 pixels: ' + str(data[:10]))
    # print('\n')    

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()