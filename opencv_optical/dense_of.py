import numpy as np
import cv2 as cv
from PIL import Image
import glob
import os


def load_image_gray(imfile):
    img = cv.imread(imfile,cv.IMREAD_GRAYSCALE)
    return img

def load_image_rgb(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[:,:,:3]
    img =cv.cvtColor(img,cv.COLOR_RGB2BGR)
    #print(img.shape)
    img =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img

def load_image(imfile,mode):
    if mode:
        img = load_image_rgb(imfile)
    else:
        img = load_image_gray(imfile)
    return img

def save_res(bgr, save,i):
    cv.imwrite(os.path.join(save,str(i).zfill(5)+'.png'), bgr)

def run(data,save,mode):
    images = glob.glob(os.path.join(data, '*.png'))
    images = sorted(images)
    #print(images)
    i = 0
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1,mode)
            image2 = load_image(imfile2,mode)
            hsv = np.zeros((376,672,3))
            #print(hsv.shape)
            hsv[...,1]=255
            flow = cv.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 7, 5, 7, 1.5, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            hsv = np.float32(hsv)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            save_res(bgr,save,i)
            i = i+1


if __name__ == "__main__":
    data_path = "../images/small_img"
    save_path = "result/cout_of_res"
    run(data_path, save_path,True)#mode True = rgb, False = gray