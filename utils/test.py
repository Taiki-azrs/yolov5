import math
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
# from utils.metrics import bbox_ioa
class hoge:
    def __init__(self,img_size=640) -> None:
        self.img_size=640

    # def __call__(self, image,bboxes,class_labels):
    def __call__(self, **params):
        xmin=int(params['bboxes'][0][0]*self.img_size)
        ymin=int(params['bboxes'][0][1]*self.img_size)
        xmax=int(params['bboxes'][0][2]*self.img_size)
        ymax=int(params['bboxes'][0][3]*self.img_size)
        bbox_w = xmax-xmin
        bbox_h = ymax-ymin
        img = params['image']
        bbox_img = img[0,ymin:ymax,xmin:xmax].copy()
        rate=random.random()
        resized_size=int(self.img_size*rate)
        bbox_img = cv2.resize(bbox_img, dsize=(resized_size,resized_size))
        pos_x = random.random()
        pos_y = random.random()
        img[
            0,
            int(pos_x*resized_size) : int(pos_x*resized_size + resized_size),
            int(pos_y*resized_size) : int(pos_y*resized_size + resized_size)
        ]=bbox_img[:]
        params['bboxes'][0]=(pos_x,pos_y,pos_x+rate,pos_y+rate)

        return params
    
class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        # prefix = colorstr('albumentations: ')
        try:
            import albumentations as A
            # check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            # T = [
            #     A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
            #     A.RGBShift(p=0.6),
            #     A.ChannelShuffle(p=0.6),
            #     A.Blur(p=0.01),
            #     A.MedianBlur(p=0.01),
            #     A.ToGray(p=0.05),
            #     A.CLAHE(p=0.01),
            #     A.RandomBrightnessContrast(p=0.0),
            #     A.RandomGamma(p=0.0),
            #     A.ShiftScaleRotate(shift_limit=0.5,scale_limit=0,rotate_limit=0,p=0.7),
            #     A.Downscale(p=0.1),
            #     A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            T = [
                # A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                hoge(img_size=640)
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            # LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        # except Exception as e:
            # LOGGER.info(f'{prefix}{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels

def main():

    albumentations = Albumentations(size=640)
    l = np.zeros((1,5))
    l[0,0]=0.0
    [(0.329, 0.1725, 0.7070000000000001, 0.8455, 0.0)]
    l[0,1]=0.518
    l[0,2]=0.509
    l[0,3]=0.378
    l[0,4]=0.673
    c = np.zeros((1,1))
    c[0,0]=0
    img = cv2.imread('test.png')
    # img = img.transpose(1,2,0)
    img = cv2.resize(img,(640,640))
    img = img[np.newaxis,:,:,:]
    print(img.shape)
    img, labels = albumentations(img,l)
    cv2.imwrite('out.png',img[0,:,:,:])
if __name__ =="__main__":
    main()
